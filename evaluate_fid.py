"""Computes the FID of a trained consistency model against the CIFAR-10 training set.

Example::

    accelerate launch --num_processes 8 evaluate_fid.py \\
        --ckpt outputs/ict/checkpoints/last.pt --sigmas 80 0.821
"""

import argparse

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object, set_seed

from common import FIDEvaluator, cifar10, load_model, sample_images


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="EMA weights (.pth) or a training checkpoint (last.pt).",
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Per-process batch size."
    )
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[80.0],
        help="Decreasing noise levels; use '80 0.821' for two-step iCT sampling.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-npz",
        help="Also save the generated images as a uint8 NHWC array to this path.",
    )
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device
    set_seed(args.seed, device_specific=True)

    model = load_model(args.ckpt).to(device)
    with accelerator.main_process_first():
        reference = cifar10(args.data_dir, train=True, normalize=False)
    evaluator = FIDEvaluator(
        accelerator,
        reference,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    generated = []

    def generate(num_images: int):
        images = sample_images(model, num_images, args.sigmas, device)
        if args.save_npz:
            uint8 = ((images + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
            generated.append(uint8.permute(0, 2, 3, 1).cpu().numpy())
        return images

    fid = evaluator(generate)
    accelerator.print(
        f"FID ({args.num_samples} samples, sigmas={args.sigmas}): {fid:.4f}"
    )

    if args.save_npz:
        arrays = gather_object([np.concatenate(generated)])
        if accelerator.is_main_process:
            np.savez(args.save_npz, arr_0=np.concatenate(arrays))
            print(f"Saved generated images to {args.save_npz}")


if __name__ == "__main__":
    main()
