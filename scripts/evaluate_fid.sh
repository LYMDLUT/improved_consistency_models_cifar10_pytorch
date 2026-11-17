#!/usr/bin/env bash
# One-step and two-step FID-50k of a trained iCT model.
# Usage: scripts/evaluate_fid.sh outputs/ict/checkpoints/last.pt
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=${1:?"usage: $0 CHECKPOINT"}
accelerate launch --num_processes 8 --main_process_port 8979 \
    evaluate_fid.py --ckpt "$CKPT" --sigmas 80
accelerate launch --num_processes 8 --main_process_port 8979 \
    evaluate_fid.py --ckpt "$CKPT" --sigmas 80 0.821
