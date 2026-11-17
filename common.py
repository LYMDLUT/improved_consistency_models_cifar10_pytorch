"""Helpers shared by the training, sampling and evaluation scripts."""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Optional, Union

import torch
from accelerate import Accelerator
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets, transforms

from configs.default_cifar10_configs import get_config
from consistency_models import ConsistencySamplingAndEditing
from models.ncsnpp import NCSNpp
from utils import fork_rng

IMAGE_SHAPE = (3, 32, 32)
SIGMA_MIN = 0.002
SIGMA_DATA = 0.5

PathLike = Union[str, os.PathLike]


def build_model(
    dropout: Optional[float] = None, fourier_scale: Optional[float] = None
) -> NCSNpp:
    """Builds the NCSN++ network, optionally overriding dropout and Fourier scale."""
    config = get_config()
    if dropout is not None:
        config.model.dropout = dropout
    if fourier_scale is not None:
        config.model.fourier_scale = fourier_scale
    return NCSNpp(config)


def extract_state_dict(checkpoint: dict, key: str = "ema") -> dict[str, Tensor]:
    """Returns model weights from either a plain state dict or a training checkpoint.

    Prefixes added by wrappers such as DDP (``module.``) are removed.
    """
    if all(isinstance(v, Tensor) for v in checkpoint.values()):
        state_dict = checkpoint
    else:
        state_dict = checkpoint[key]
    for prefix in ("module.", "model_ema."):
        if all(k.startswith(prefix) for k in state_dict):
            state_dict = {k.removeprefix(prefix): v for k, v in state_dict.items()}
    return state_dict


def load_model(path: PathLike, key: str = "ema") -> NCSNpp:
    """Builds NCSN++ and loads weights saved by ``train.py`` or the original scripts."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model()
    model.load_state_dict(extract_state_dict(checkpoint, key))
    return model.eval().requires_grad_(False)


def cifar10(
    root: PathLike,
    train: bool = True,
    random_flip: bool = False,
    normalize: bool = True,
) -> Dataset:
    """CIFAR-10 with images in ``[-1, 1]`` if ``normalize`` else in ``[0, 1]``."""
    transform = [transforms.RandomHorizontalFlip()] if random_flip else []
    transform.append(transforms.ToTensor())
    if normalize:
        transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return datasets.CIFAR10(
        root=root, train=train, transform=transforms.Compose(transform), download=True
    )


@torch.no_grad()
def sample_images(
    model: nn.Module,
    num_images: int,
    sigmas: Sequence[float],
    device: torch.device,
    seed: Optional[int] = None,
) -> Tensor:
    """Generates images in ``[-1, 1]`` with one- or multi-step consistency sampling.

    With a ``seed`` the global RNG state is left untouched, so it can be used inside a
    training loop without affecting the training noise.
    """
    sampler = ConsistencySamplingAndEditing(sigma_min=SIGMA_MIN, sigma_data=SIGMA_DATA)
    reference = torch.zeros((num_images, *IMAGE_SHAPE), device=device)
    if seed is None:
        return sampler(model, reference, sigmas=list(sigmas), clip_denoised=True)
    with fork_rng(device):
        torch.manual_seed(seed)
        return sampler(model, reference, sigmas=list(sigmas), clip_denoised=True)


class FIDEvaluator:
    """FID against a reference dataset, with all work sharded across processes.

    Each process extracts Inception features for its shard of the reference images
    once, and for its share of the generated images on every call; torchmetrics sums
    the feature statistics of all processes in ``compute``.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        reference_dataset: Dataset,
        num_samples: int = 50_000,
        batch_size: int = 256,
        num_workers: int = 4,
    ) -> None:
        self.accelerator = accelerator
        self.num_samples = num_samples
        self.batch_size = batch_size

        rank, world_size = accelerator.process_index, accelerator.num_processes
        with accelerator.main_process_first():
            self.fid = FrechetInceptionDistance(
                reset_real_features=False, normalize=True
            ).to(accelerator.device)

        shard = Subset(
            reference_dataset, range(rank, len(reference_dataset), world_size)
        )
        loader = DataLoader(shard, batch_size=batch_size, num_workers=num_workers)
        for images, _ in loader:
            self.fid.update(images.to(accelerator.device), real=True)

    @torch.no_grad()
    def __call__(self, generate_fn: Callable[[int], Tensor]) -> float:
        """Computes FID for ``num_samples`` images in ``[-1, 1]`` from ``generate_fn``.

        Must be called on every process.
        """
        rank, world_size = (
            self.accelerator.process_index,
            self.accelerator.num_processes,
        )
        num_local = len(range(rank, self.num_samples, world_size))
        for start in range(0, num_local, self.batch_size):
            images = generate_fn(min(self.batch_size, num_local - start))
            self.fid.update(((images + 1) / 2).clamp(0, 1), real=False)
        value = float(self.fid.compute())
        self.fid.reset()
        return value


def save_checkpoint(path: PathLike, state: dict) -> None:
    """Saves atomically, so an interrupted write never corrupts an existing file."""
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)
