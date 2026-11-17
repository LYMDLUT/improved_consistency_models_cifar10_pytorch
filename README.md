# Improved Consistency Training (iCT) on CIFAR-10 in PyTorch

A PyTorch re-implementation of [Improved Techniques for Training Consistency Models](https://arxiv.org/abs/2310.14189)
(iCT, Song & Dhariwal, 2023) on CIFAR-10 by @Buntender and @LYMDLUT. The original
consistency training (CT) of [Consistency Models](https://arxiv.org/abs/2303.01469) is
included as a baseline. The network is NCSN++ from
[score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch).

## Setup

```bash
pip install -r requirements.txt
```

CIFAR-10 is downloaded to `--data-dir` (default `./data`) on first use.

The FIR up/down-sampling op of NCSN++ compiles a CUDA extension on first use and falls
back to an equivalent pure PyTorch implementation if compilation fails. Set
`UPFIRDN2D_BACKEND=native` to skip compilation, or `UPFIRDN2D_BACKEND=cuda` to make a
failed compilation an error.

## Training

```bash
bash scripts/train_ict.sh          # iCT, paper hyper-parameters
bash scripts/train_ict_200e.sh     # short 200-epoch iCT run
bash scripts/train_ct.sh           # original CT with LPIPS
```

The scripts wrap `accelerate launch --num_processes 8 train.py --method {ict,ct}`; extra
arguments are forwarded, and `python train.py --help` lists all options. `--batch-size`
is per process, so 8 processes give the global batch size of 1024 used in the paper.
Defaults that depend on the method:

| Option | `ict` | `ct` |
| --- | --- | --- |
| `--lr` (RAdam) | 1e-4 | 2e-4 |
| `--initial-timesteps` / `--final-timesteps` | 10 / 1280 | 2 / 150 |
| `--ema-decay` (student EMA used for evaluation) | 0.99993 | 0.9999 |
| `--dropout` | 0.3 | 0.0 |
| `--fourier-scale` | 0.02 | 16.0 |
| Teacher | student itself (no EMA) | EMA, `--teacher-ema-mu0 0.9` |
| Loss | Pseudo-Huber, weighted by 1 / (σ<sub>i+1</sub> − σ<sub>i</sub>) | LPIPS (VGG + SqueezeNet) + L1 |

Each run writes to `--output-dir` (default `outputs/<method>`):

```
args.json                  # arguments of the run
metrics.jsonl              # per-epoch loss and FID
samples/epoch_XXXX.png     # one-step samples from the EMA model
checkpoints/last.pt        # full training state, resume with --resume
checkpoints/ema_XXXX.pth   # EMA weights every --snapshot-every epochs
```

## Sampling and evaluation

```bash
# Sample grid, one-step (--sigmas 80) or two-step (--sigmas 80 0.821) sampling
python sample.py --ckpt outputs/ict/checkpoints/last.pt --sigmas 80 0.821

# FID-50k against the CIFAR-10 training set
accelerate launch --num_processes 8 evaluate_fid.py --ckpt outputs/ict/checkpoints/last.pt

# Accuracy of a classifier on test images purified by the consistency model
python eval_purification.py --dae-ckpt outputs/ict/checkpoints/last.pt --clf-ckpt origin.t7 --sigma 0.5
```

`--ckpt` accepts `last.pt` as well as plain weight files, including those written by
earlier versions of this repository (e.g. `ict_ema_4096e.pth`).

## Results

FID of one-step samples, obtained with the original scripts before the fixes listed
below. The FID evaluation of those scripts was affected by some of these issues, so the
numbers are not directly comparable to runs of the current code.

| epoch | initial_timesteps | final_timesteps | ema | FID |
| --- | --- | --- | --- | --- |
| 2048 | 10 | 1280 | 0.9999 | 14.7302532 |
| 4096 | 10 | 1280 | 0.99993 | 13.039658 |
| 200 | 10 | 1280 | 0.999 | 361.96563 |
| 200 | 10 | 150 | 0.99 | 54.1811 |
| 200 | 2 | 150 | 0.99 | 40.65 |
| 200 | 2 | 150 | 0.9 | 64.6632 |
| 500 | 2 | 150 | 0.99 | 34.1635 |

<details>
<summary>FID during training of the 4096-epoch run (epoch, FID)</summary>

 - 50, FID: 398.509033203125
 - 100, FID: 296.0087585449219
 - 150, FID: 239.63021850585938
 - 200, FID: 160.6717987060547
 - 250, FID: 106.63436126708984
 - 300, FID: 88.66282653808594
 - 350, FID: 85.65467071533203
 - 400, FID: 84.96791076660156
 - 450, FID: 84.6329345703125
 - 500, FID: 84.25988006591797
 - 550, FID: 62.964683532714844
 - 600, FID: 56.208831787109375
 - 650, FID: 53.99787521362305
 - 700, FID: 53.00424575805664
 - 750, FID: 52.650970458984375
 - 800, FID: 52.970985412597656
 - 850, FID: 53.424232482910156
 - 900, FID: 54.50812911987305
 - 950, FID: 55.11760330200195
 - 1000, FID: 55.904388427734375
 - 1050, FID: 47.12823486328125
 - 1100, FID: 39.525386810302734
 - 1150, FID: 37.73923873901367
 - 1200, FID: 36.83944320678711
 - 1250, FID: 37.3828010559082
 - 1300, FID: 38.235687255859375
 - 1350, FID: 38.59758758544922
 - 1400, FID: 38.64599609375
 - 1450, FID: 39.1431999206543
 - 1500, FID: 39.39213562011719
 - 1550, FID: 35.01113510131836
 - 1600, FID: 32.2806396484375
 - 1650, FID: 31.84785270690918
 - 1700, FID: 30.873199462890625
 - 1750, FID: 29.961013793945312
 - 1800, FID: 29.411357879638672
 - 1850, FID: 28.710996627807617
 - 1900, FID: 28.08704948425293
 - 1950, FID: 28.2253360748291
 - 2000, FID: 28.125417709350586
 - 2050, FID: 27.524084091186523
 - 2100, FID: 25.96059799194336
 - 2150, FID: 26.959360122680664
 - 2200, FID: 19.976266860961914
 - 2250, FID: 18.47594451904297
 - 2300, FID: 17.9345760345459
 - 2350, FID: 18.18851089477539
 - 2400, FID: 17.921463012695312
 - 2450, FID: 17.91927719116211
 - 2500, FID: 17.469663619995117
 - 2550, FID: 17.566999435424805
 - 2600, FID: 28.899091720581055
 - 2650, FID: 19.934633255004883
 - 2700, FID: 19.609600067138672
 - 2750, FID: 30.208534240722656
 - 2800, FID: 27.900463104248047
 - 2850, FID: 25.653335571289062
 - 2900, FID: 23.310134887695312
 - 2950, FID: 19.263086318969727
 - 3000, FID: 16.308795928955078
 - 3050, FID: 14.913342475891113
 - 3100, FID: 13.603191375732422
 - 3150, FID: 13.039658546447754
 - 3200, FID: 13.350031852722168
 - 3250, FID: 13.559409141540527
 - 3300, FID: 13.257262229919434
 - 3350, FID: 14.365655899047852
 - 3400, FID: 15.250137329101562
 - 3450, FID: 16.243085861206055
 - 3500, FID: 20.085010528564453
 - 3550, FID: 21.34207534790039
 - 3600, FID: 26.837499618530273
 - 3650, FID: 28.421476364135742
 - 3700, FID: 39.8241081237793
 - 3750, FID: 42.5471076965332
 - 3800, FID: 42.17381286621094
 - 3850, FID: 37.740726470947266
 - 3900, FID: 43.5645637512207
 - 3950, FID: 55.518592834472656
 - 4000, FID: 63.49004364013672
 - 4050, FID: 56.94802474975586

</details>

## Changes from the original scripts

Fixes that change training or evaluation:

- **Pseudo-Huber loss.** The loss is now `sqrt(||x - y||² + c²) - c` over each whole
  image, as in the paper. It used to be applied to every pixel independently, with `c`
  still computed for whole images.
- **iCT hyper-parameters.** Dropout 0.3, Fourier embedding scale 0.02, and the student
  and teacher forward passes share dropout masks (Section 3.1 of the paper). The
  original scripts used dropout 0 and scale 16; pass `--dropout 0 --fourier-scale 16`
  to reproduce them.
- **EMA initialisation in multi-GPU runs.** The EMA models were copied from each
  process's own random initialisation, while DDP synchronises the student to the
  weights of process 0, so the EMA weights differed between processes. FID samples
  were drawn from these differing EMA models.
- **FID evaluation.** Every process computed reference statistics on the full training
  set and accumulated the gathered samples of all processes. Work is now sharded
  across processes, exactly `--fid-num-samples` images are generated (it was 49,152
  instead of 50,000), and the reference images are no longer randomly flipped. The
  standalone FID scripts also skipped the last batch of generated images.
- **Learning-rate warmup.** `--warmup-steps` counts optimizer steps. The scheduler was
  previously prepared by Accelerate, which steps it once per process on every
  optimizer step, shortening the warmup by the number of processes.
- `improved_timesteps_schedule` returned a float instead of an int, the sampling
  progress bar was shown only with `verbose=False`, single-GPU runs failed on an
  unconditional `torch.distributed.barrier()`, and purification silently skipped
  missing weights (`strict=False`).

Code organisation:

- `ict.py`, `ict_200e.py` and `ct.py` are merged into `train.py`, with checkpoints
  that can be resumed; `sampling.py`, `fid/` and `purify_clean_clscm.py` became
  `sample.py`, `evaluate_fid.py` and `eval_purification.py`. Hard-coded paths are
  replaced by command-line arguments.
- The CUDA op is no longer compiled at import time, the unused `fused_act` op and the
  duplicated `models/op_install` are removed, as are helpers in `models/utils.py` that
  referenced the missing `sde_lib`.
- Added `requirements.txt`, unit tests and a `ruff` configuration.

## Development

```bash
pip install pytest ruff
pytest
ruff check . && ruff format --check .
```

## Acknowledgements

The consistency training and sampling code is adapted from
[Kinyugo/consistency_models](https://github.com/Kinyugo/consistency_models), and the
NCSN++ model, including the `upfirdn2d` op, comes from
[score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch).
