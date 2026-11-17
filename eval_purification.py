"""Classifier accuracy on CIFAR-10 test images purified by a consistency model.

Every test image is perturbed with Gaussian noise of standard deviation ``--sigma``
and denoised with a single consistency-model step before being classified.
``--sigma 0`` skips purification and reports the clean accuracy.

Example::

    python eval_purification.py --dae-ckpt outputs/ict/checkpoints/last.pt \\
        --clf-ckpt origin.t7 --sigma 0.5
"""

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from common import SIGMA_DATA, SIGMA_MIN, cifar10, load_model
from consistency_models import model_forward_wrapper
from networks.wide_resnet import WideResnet_cifar10

CLASSIFIER_MEAN = (0.4914, 0.4822, 0.4465)
CLASSIFIER_STD = (0.2023, 0.1994, 0.2010)


def load_classifier(path: str) -> nn.Module:
    """Loads a WideResNet-28-10 pickled as ``{'net': model}`` or as a state dict."""
    # Pickled modules cannot be loaded with weights_only=True; only use trusted files.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    net = (
        checkpoint["net"]
        if isinstance(checkpoint, dict) and "net" in checkpoint
        else checkpoint
    )
    if isinstance(net, nn.DataParallel):
        net = net.module
    if isinstance(net, dict):
        model = WideResnet_cifar10()
        model.load_state_dict({k.removeprefix("module."): v for k, v in net.items()})
        net = model
    return net.eval()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dae-ckpt", required=True, help="Consistency model weights.")
    parser.add_argument(
        "--clf-ckpt", default="./origin.t7", help="CIFAR-10 classifier."
    )
    parser.add_argument(
        "--sigma", type=float, default=0.5, help="Purification noise level."
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    denoiser = load_model(args.dae_ckpt).to(device)
    classifier = load_classifier(args.clf_ckpt).to(device)
    normalize = transforms.Normalize(CLASSIFIER_MEAN, CLASSIFIER_STD)

    test_set = cifar10(args.data_dir, train=False)
    loader = DataLoader(
        test_set, batch_size=args.batch_size, num_workers=args.num_workers
    )

    torch.manual_seed(args.seed)
    correct = total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images, labels = images.to(device), labels.to(device)
            if args.sigma > 0:
                sigma = torch.full((images.shape[0],), args.sigma, device=device)
                noisy = images + args.sigma * torch.randn_like(images)
                images = model_forward_wrapper(
                    denoiser, noisy, sigma, SIGMA_DATA, SIGMA_MIN
                )
                images = images.clamp(-1.0, 1.0)
            logits = classifier(normalize((images + 1) / 2))
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.numel()

    print(f"Accuracy (sigma={args.sigma}): {correct / total:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
