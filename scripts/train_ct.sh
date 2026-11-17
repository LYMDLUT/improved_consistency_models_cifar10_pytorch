#!/usr/bin/env bash
# Original consistency training (CT) with an EMA teacher and LPIPS loss.
set -euo pipefail
cd "$(dirname "$0")/.."

accelerate launch --num_processes 8 --main_process_port 4066 \
    train.py --method ct --output-dir outputs/ct "$@"
