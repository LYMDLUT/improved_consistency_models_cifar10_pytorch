"""Trains a consistency model on CIFAR-10 with iCT (default) or the original CT.

Launch with ``accelerate launch --num_processes 8 train.py [--method ict|ct] ...``.
``--batch-size`` is per process, so 8 processes x 128 gives the global batch size
of 1024 used in the iCT paper.
"""

import argparse
import json
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.utils import save_image

from common import FIDEvaluator, build_model, cifar10, sample_images, save_checkpoint
from consistency_models import (
    ConsistencyTraining,
    ImprovedConsistencyTraining,
    ema_decay_rate_schedule,
    pseudo_huber_loss,
)
from utils import update_ema_model_

# Defaults that differ between iCT (Song & Dhariwal, 2023) and CT (Song et al., 2023).
# Arguments left unset on the command line fall back to these values.
METHOD_DEFAULTS = {
    "ict": dict(
        lr=1e-4,
        ema_decay=0.99993,
        initial_timesteps=10,
        final_timesteps=1280,
        dropout=0.3,
        fourier_scale=0.02,
        fid_every=10,
    ),
    "ct": dict(
        lr=2e-4,
        ema_decay=0.9999,
        initial_timesteps=2,
        final_timesteps=150,
        dropout=0.0,
        fourier_scale=16.0,
        fid_every=5,
    ),
}
SAMPLING_SIGMAS = [80.0]
NUM_PREVIEW_IMAGES = 64


class PerceptualLoss(nn.Module):
    """LPIPS (summed over backbones) plus a weighted L1 term, as used for CT."""

    def __init__(
        self, net_types: Sequence[str] = ("vgg", "squeeze"), l1_weight: float = 1.0
    ):
        super().__init__()
        self.lpips_losses = nn.ModuleList(
            LearnedPerceptualImagePatchSimilarity(net_type)
            for net_type in sorted(net_types)
        )
        self.lpips_losses.requires_grad_(False)
        self.l1_weight = l1_weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        upscaled_input = F.interpolate(input, (224, 224), mode="bilinear")
        upscaled_target = F.interpolate(target, (224, 224), mode="bilinear")
        lpips_loss = sum(
            lpips(upscaled_input, upscaled_target) for lpips in self.lpips_losses
        )
        return lpips_loss + self.l1_weight * F.l1_loss(input, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--method", choices=sorted(METHOD_DEFAULTS), default="ict")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", help="Defaults to ./outputs/<method>.")
    parser.add_argument("--epochs", type=int, default=4096)
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Per-process batch size."
    )
    parser.add_argument("--lr", type=float)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Linear LR warmup in optimizer steps.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        help="EMA decay of the student weights used for evaluation.",
    )
    parser.add_argument(
        "--initial-timesteps", type=int, help="s0 of the discretization curriculum."
    )
    parser.add_argument(
        "--final-timesteps", type=int, help="s1 of the discretization curriculum."
    )
    parser.add_argument("--dropout", type=float)
    parser.add_argument(
        "--fourier-scale", type=float, help="Scale of the Fourier noise embedding."
    )
    parser.add_argument(
        "--teacher-ema-mu0",
        type=float,
        default=0.9,
        help="CT only: initial teacher EMA decay.",
    )
    parser.add_argument(
        "--lpips-nets",
        nargs="+",
        choices=["alex", "squeeze", "vgg"],
        default=["vgg", "squeeze"],
        help="CT only: LPIPS backbones.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-every", type=int, default=50, help="Steps between loss logs."
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Epochs between sample grids (0 disables).",
    )
    parser.add_argument(
        "--ckpt-every", type=int, default=10, help="Epochs between overwriting last.pt."
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=50,
        help="Epochs between keeping a copy of the EMA weights (0 disables).",
    )
    parser.add_argument(
        "--fid-every", type=int, help="Epochs between FID evaluations (0 disables)."
    )
    parser.add_argument("--fid-num-samples", type=int, default=50_000)
    parser.add_argument("--fid-batch-size", type=int, default=256)
    parser.add_argument("--resume", help="Path to a last.pt checkpoint to resume from.")
    parser.add_argument(
        "--max-steps", type=int, help="Stop after this many steps (debugging)."
    )
    args = parser.parse_args()

    for key, value in METHOD_DEFAULTS[args.method].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.output_dir is None:
        args.output_dir = f"./outputs/{args.method}"
    return args


def is_due(epoch: int, every: int, is_last: bool) -> bool:
    return every > 0 and ((epoch + 1) % every == 0 or is_last)


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device
    torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    sample_dir, ckpt_dir = output_dir / "samples", output_dir / "checkpoints"
    if accelerator.is_main_process:
        sample_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    # The same seed on every process makes the EMA copies start from the same weights
    # that DDP broadcasts to the student.
    set_seed(args.seed)
    model = build_model(dropout=args.dropout, fourier_scale=args.fourier_scale)
    ema_model = deepcopy(model).requires_grad_(False).eval()
    teacher_model = (
        deepcopy(model).requires_grad_(False) if args.method == "ct" else None
    )

    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(args.warmup_steps, 1))
    )

    with accelerator.main_process_first():
        train_set = cifar10(args.data_dir, train=True, random_flip=True)
        fid_reference = cifar10(args.data_dir, train=True, normalize=False)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    # The scheduler is stepped manually so that warmup counts optimizer steps
    # regardless of the number of processes.
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    ema_model.to(device)
    if teacher_model is not None:
        teacher_model.to(device)

    step, start_epoch = 0, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        accelerator.unwrap_model(model).load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema"])
        if teacher_model is not None:
            teacher_model.load_state_dict(checkpoint["teacher"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step, start_epoch = checkpoint["step"], checkpoint["epoch"]
        accelerator.print(
            f"Resumed from {args.resume} at epoch {start_epoch}, step {step}"
        )

    # Each process must draw different noise from here on.
    set_seed(args.seed + step, device_specific=True)

    if args.method == "ict":
        trainer = ImprovedConsistencyTraining(
            initial_timesteps=args.initial_timesteps,
            final_timesteps=args.final_timesteps,
        )
    else:
        trainer = ConsistencyTraining(
            initial_timesteps=args.initial_timesteps,
            final_timesteps=args.final_timesteps,
        )
        with accelerator.main_process_first():
            lpips = PerceptualLoss(net_types=args.lpips_nets).to(device)

    fid_evaluator = None
    if args.fid_every > 0:
        fid_evaluator = FIDEvaluator(
            accelerator,
            fid_reference,
            num_samples=args.fid_num_samples,
            batch_size=args.fid_batch_size,
            num_workers=args.num_workers,
        )

    total_steps = args.epochs * len(train_loader)
    num_params = sum(p.numel() for p in ema_model.parameters())
    accelerator.print(
        f"method={args.method} params={num_params / 1e6:.1f}M "
        f"global_batch={args.batch_size * accelerator.num_processes} "
        f"steps_per_epoch={len(train_loader)} total_steps={total_steps}"
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_sum, epoch_steps = torch.zeros((), device=device), 0
        for images, _ in train_loader:
            if args.method == "ict":
                output = trainer(model, images, step, total_steps)
                loss = pseudo_huber_loss(output.predicted, output.target)
                loss = (loss * output.loss_weights).mean()
            else:
                output = trainer(model, teacher_model, images, step, total_steps)
                loss = lpips(output.predicted.clamp(-1, 1), output.target.clamp(-1, 1))

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            student = accelerator.unwrap_model(model)
            if teacher_model is not None:
                teacher_decay = ema_decay_rate_schedule(
                    output.num_timesteps, args.teacher_ema_mu0, args.initial_timesteps
                )
                update_ema_model_(teacher_model, student, teacher_decay)
            update_ema_model_(ema_model, student, args.ema_decay)

            step += 1
            epoch_steps += 1
            loss_sum += loss.detach()
            if step % args.log_every == 0:
                mean_loss = accelerator.reduce(loss.detach(), "mean").item()
                accelerator.print(
                    f"epoch {epoch + 1}/{args.epochs} step {step}/{total_steps} "
                    f"loss {mean_loss:.4f} N(k) {output.num_timesteps} "
                    f"lr {scheduler.get_last_lr()[0]:.2e}"
                )
            if args.max_steps is not None and step >= args.max_steps:
                break

        is_last = epoch + 1 == args.epochs or (
            args.max_steps is not None and step >= args.max_steps
        )
        record = {
            "epoch": epoch + 1,
            "step": step,
            "loss": accelerator.reduce(loss_sum, "mean").item() / max(epoch_steps, 1),
        }

        if accelerator.is_main_process and is_due(epoch, args.sample_every, is_last):
            images = sample_images(
                ema_model, NUM_PREVIEW_IMAGES, SAMPLING_SIGMAS, device, seed=args.seed
            )
            save_image(
                (images + 1) / 2, sample_dir / f"epoch_{epoch + 1:04d}.png", nrow=8
            )

        if fid_evaluator is not None and is_due(epoch, args.fid_every, is_last):
            record["fid"] = fid_evaluator(
                lambda n: sample_images(ema_model, n, SAMPLING_SIGMAS, device)
            )

        accelerator.print(json.dumps(record))
        if accelerator.is_main_process:
            with open(output_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")

            if is_due(epoch, args.ckpt_every, is_last):
                state = {
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "ema": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "epoch": epoch + 1,
                    "args": vars(args),
                }
                if teacher_model is not None:
                    state["teacher"] = teacher_model.state_dict()
                save_checkpoint(ckpt_dir / "last.pt", state)
            if args.snapshot_every > 0 and (epoch + 1) % args.snapshot_every == 0:
                torch.save(
                    ema_model.state_dict(), ckpt_dir / f"ema_{epoch + 1:04d}.pth"
                )

        if is_last:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
