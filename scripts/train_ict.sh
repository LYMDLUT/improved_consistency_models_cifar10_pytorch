#!/usr/bin/env bash
# iCT on CIFAR-10 with the paper's hyper-parameters (global batch size 8 x 128 = 1024).
set -euo pipefail
cd "$(dirname "$0")/.."

accelerate launch --num_processes 8 --main_process_port 4066 \
    train.py --method ict --output-dir outputs/ict "$@"
