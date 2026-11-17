"""Saves a grid of samples from a trained consistency model.

Example::

    python sample.py --ckpt outputs/ict/checkpoints/last.pt --sigmas 80 0.821
"""

import argparse

import torch
from torchvision.utils import save_image

from common import load_model, sample_images


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="EMA weights (.pth) or a training checkpoint (last.pt).",
    )
    parser.add_argument("--output", default="samples.png")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[80.0],
        help="Decreasing noise levels; use '80 0.821' for two-step iCT sampling.",
    )
    parser.add_argument("--nrow", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.ckpt).to(device)
    torch.manual_seed(args.seed)
    images = sample_images(model, args.num_samples, args.sigmas, device)
    save_image((images + 1) / 2, args.output, nrow=args.nrow)
    print(f"Saved {args.num_samples} samples to {args.output}")


if __name__ == "__main__":
    main()
