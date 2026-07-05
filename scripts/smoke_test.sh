#!/usr/bin/env bash
# Linux/Mac smoke test: full pipeline on synthetic data, CPU, ~1-2 min.
set -e
python -m csarec.train --config configs/default.yaml \
  --data.source synthetic --device cpu \
  --data.synth_users 400 --data.synth_items 300 \
  --train.epochs 20 --train.eval_every 5 --output.tag smoke
