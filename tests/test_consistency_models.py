import math

import pytest
import torch
from torch import nn

from common import build_model, extract_state_dict, load_model, sample_images
from consistency_models import (
    ConsistencyTraining,
    ImprovedConsistencyTraining,
    improved_timesteps_schedule,
    karras_schedule,
    lognormal_timestep_distribution,
    pseudo_huber_loss,
    timesteps_schedule,
)
from models.up_or_down_sampling import downsample_2d, upsample_2d
from utils import update_ema_model_

CPU = torch.device("cpu")


class DropoutProbe(nn.Module):
    """Records the dropout mask of every forward pass."""

    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.masks = []

    def forward(self, x, sigma):
        out = self.dropout(torch.ones_like(x))
        self.masks.append(out != 0)
        return out


def test_improved_timesteps_schedule_doubles_until_final():
    total = 8000
    values = [
        improved_timesteps_schedule(k, total, 10, 1280)
        for k in range(0, total + 1, 1000)
    ]
    assert all(isinstance(v, int) for v in values)
    # K' = floor(K / (log2(1280 / 10) + 1)) = 1000.
    assert values == [11, 21, 41, 81, 161, 321, 641, 1281, 1281]


def test_timesteps_schedule_endpoints():
    assert timesteps_schedule(0, 100, 2, 150) == 2
    assert timesteps_schedule(100, 100, 2, 150) == 151


def test_karras_schedule_endpoints():
    sigmas = karras_schedule(11)
    assert sigmas[0].item() == pytest.approx(0.002)
    assert sigmas[-1].item() == pytest.approx(80.0)
    assert torch.all(sigmas[1:] > sigmas[:-1])


def test_lognormal_timesteps_index_adjacent_pairs():
    sigmas = karras_schedule(11)
    timesteps = lognormal_timestep_distribution(1000, sigmas)
    assert timesteps.min() >= 0
    assert timesteps.max() <= len(sigmas) - 2


def test_pseudo_huber_uses_per_sample_l2_norm():
    x, y = torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)
    c = 0.00054 * math.sqrt(3 * 32 * 32)
    expected = torch.sqrt((x - y).pow(2).sum(dim=(1, 2, 3)) + c**2) - c

    loss = pseudo_huber_loss(x, y)

    assert loss.shape == (4,)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(pseudo_huber_loss(x, x), torch.zeros(4))


def test_ict_student_and_teacher_share_dropout_masks():
    model = DropoutProbe().train()
    ict = ImprovedConsistencyTraining()

    output = ict(model, torch.randn(8, 3, 32, 32), 0, 100)
    ict(model, torch.randn(8, 3, 32, 32), 1, 100)

    assert output.loss_weights.shape == (8,)
    assert torch.equal(model.masks[0], model.masks[1])
    assert not torch.equal(model.masks[0], model.masks[2])


def test_ct_student_and_teacher_share_dropout_masks():
    student, teacher = DropoutProbe().train(), DropoutProbe().train()
    ConsistencyTraining()(student, teacher, torch.randn(8, 3, 32, 32), 0, 100)
    assert torch.equal(student.masks[0], teacher.masks[0])


def test_update_ema_model():
    ema, online = nn.Linear(2, 2), nn.Linear(2, 2)
    expected = [
        0.9 * e.detach() + 0.1 * o.detach()
        for e, o in zip(ema.parameters(), online.parameters())
    ]
    update_ema_model_(ema, online, 0.9)
    for param, value in zip(ema.parameters(), expected):
        torch.testing.assert_close(param.detach(), value)


def test_fir_resampling_preserves_constant_images():
    x = torch.ones(2, 4, 8, 8)
    up = upsample_2d(x, k=[1, 3, 3, 1])
    down = downsample_2d(x, k=[1, 3, 3, 1])
    assert up.shape == (2, 4, 16, 16)
    assert down.shape == (2, 4, 4, 4)
    # Borders are affected by zero padding.
    torch.testing.assert_close(up[..., 2:-2, 2:-2], torch.ones(2, 4, 12, 12))
    torch.testing.assert_close(down[..., 1:-1, 1:-1], torch.ones(2, 4, 2, 2))


def test_extract_state_dict_handles_wrappers_and_checkpoints():
    assert list(extract_state_dict({"module.a": torch.zeros(1)})) == ["a"]
    checkpoint = {
        "ema": {"a": torch.zeros(1)},
        "model": {"b": torch.zeros(1)},
        "step": 3,
    }
    assert list(extract_state_dict(checkpoint)) == ["a"]
    assert list(extract_state_dict(checkpoint, key="model")) == ["b"]


@pytest.fixture(scope="module")
def ncsnpp():
    torch.manual_seed(0)
    return build_model(dropout=0.3, fourier_scale=0.02)


def test_ncsnpp_ict_step_produces_gradients(ncsnpp):
    ncsnpp.train()
    x = torch.randn(2, 3, 32, 32)

    output = ImprovedConsistencyTraining()(ncsnpp, x, 0, 10)
    loss = (
        pseudo_huber_loss(output.predicted, output.target) * output.loss_weights
    ).mean()
    loss.backward()

    assert output.predicted.shape == x.shape
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in ncsnpp.parameters() if p.requires_grad)
    ncsnpp.zero_grad(set_to_none=True)


def test_seeded_sampling_is_reproducible_and_keeps_rng_state(ncsnpp):
    ncsnpp.eval()
    rng_state = torch.get_rng_state()

    images = sample_images(ncsnpp, 2, [80.0, 0.821], CPU, seed=0)
    again = sample_images(ncsnpp, 2, [80.0, 0.821], CPU, seed=0)

    assert images.shape == (2, 3, 32, 32)
    assert images.min() >= -1 and images.max() <= 1
    assert torch.equal(images, again)
    assert torch.equal(rng_state, torch.get_rng_state())


def test_load_model_roundtrip(ncsnpp, tmp_path):
    path = tmp_path / "ema.pth"
    torch.save({f"module.{k}": v for k, v in ncsnpp.state_dict().items()}, path)
    loaded = load_model(path)
    for expected, actual in zip(
        ncsnpp.state_dict().values(), loaded.state_dict().values()
    ):
        torch.testing.assert_close(actual, expected)
