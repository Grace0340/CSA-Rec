# Windows smoke test: runs the full pipeline on synthetic data on CPU in ~1-2 min.
# From the csa-rec/ directory:  ./scripts/smoke_test.ps1
python -m csarec.train --config configs/default.yaml `
  --data.source synthetic --device cpu `
  --data.synth_users 400 --data.synth_items 300 `
  --train.epochs 20 --train.eval_every 5 --output.tag smoke
