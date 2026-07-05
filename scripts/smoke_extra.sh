#!/usr/bin/env bash
# Fast CPU smoke test for the NEW experiment code paths (learned baselines,
# temporal split, beyond-accuracy metrics). Run this FIRST, before spending GPU
# hours, to confirm every new branch executes end to end on synthetic data.
#   bash scripts/smoke_extra.sh
set -e
CFG=configs/default.yaml
COMMON="--config $CFG --data.source synthetic --device cpu --train.epochs 5 --train.baseline_head_epochs 5"

echo "== learned cold-start baselines =="
for m in content2emb dropoutnet clcrec; do
  python -m csarec.train $COMMON --method $m --output.tag smoke_$m
done

echo "== temporal cold split =="
python -m csarec.train $COMMON --data.cold_split temporal --output.tag smoke_temporal

echo "== full model (beyond-accuracy metrics should appear in metrics.json) =="
python -m csarec.train $COMMON --output.tag smoke_full

echo "== check beyond block is populated =="
python - <<'PY'
import json
d = json.load(open("runs/smoke_full/metrics.json"))
print("beyond keys:", list(d.get("beyond", {}).get("cold", {}).keys()))
assert d.get("beyond", {}).get("cold"), "beyond metrics missing!"
print("SMOKE OK")
PY
