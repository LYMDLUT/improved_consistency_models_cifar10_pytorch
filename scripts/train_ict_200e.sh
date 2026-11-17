#!/usr/bin/env bash
# Short 200-epoch iCT run, replacing the former ict_200e.py.
set -euo pipefail
cd "$(dirname "$0")/.."

accelerate launch --num_processes 8 --main_process_port 4066 \
    train.py --method ict --output-dir outputs/ict_200e \
    --epochs 200 --lr 2e-4 --warmup-steps 20 \
    --initial-timesteps 2 --final-timesteps 150 \
    --ema-decay 0.99 --fid-every 5 "$@"
