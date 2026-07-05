#!/usr/bin/env bash
# Full experiment matrix for CSA-Rec. Designed to be launched in the background:
#   nohup bash scripts/run_all.sh > run_all.log 2>&1 &
#   tail -f run_all.log
#
# Sections can be trimmed by editing CATS / SEEDS below.
set -u
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-~/autodl-tmp/hf_cache}

CFG=configs/default.yaml
SEEDS="2026 2027 2028"
CATS="Video_Games Baby_Products Toys_and_Games"   # add Beauty_and_Personal_Care last (heavy)

run() {  # run() <tag> <extra-args...>
  local tag=$1; shift
  if [ -f "runs/$tag/metrics.json" ]; then
    echo "[skip] $tag already done"; return
  fi
  echo "===== $(date +%H:%M:%S) RUN $tag ====="
  python -m csarec.train --config $CFG --semantic.method sbert --device cuda \
      --data.source amazon "$@" --output.tag "$tag" || echo "[FAIL] $tag"
}

for cat in $CATS; do
  python scripts/download_amazon.py --category "$cat" --root ./data || true
  BASE="--data.category $cat"
  for seed in $SEEDS; do
    run ${cat}_full_s${seed}       $BASE --seed $seed
    run ${cat}_ctrlcf_s${seed}     $BASE --seed $seed --model.use_semantic false
    run ${cat}_nocsia_s${seed}     $BASE --seed $seed --model.use_csia false
    run ${cat}_nocpg_s${seed}      $BASE --seed $seed --model.use_cpg false
    run ${cat}_nofusion_s${seed}   $BASE --seed $seed --model.use_fusion false
    run ${cat}_contentknn_s${seed} $BASE --seed $seed --baseline content_knn
    run ${cat}_pop_s${seed}        $BASE --seed $seed --baseline pop
    run ${cat}_random_s${seed}     $BASE --seed $seed --baseline random
  done
done

# ---- structural ablations (Video_Games, single seed) ----
VG="--data.category Video_Games --seed 2026"
run Video_Games_saalinear_s2026 $VG --model.saa_type linear
run Video_Games_mf_s2026        $VG --model.backbone mf
for r in 8 16 32 64; do
  run Video_Games_rank${r}_s2026 $VG --model.saa_rank $r
done

# ---- cold-ratio severity sweep (Video_Games, single seed) ----
for cr in 0.05 0.1 0.2 0.3; do
  run Video_Games_full_cr${cr}_s2026   --data.category Video_Games --seed 2026 --data.cold_ratio $cr
  run Video_Games_ctrlcf_cr${cr}_s2026 --data.category Video_Games --seed 2026 --data.cold_ratio $cr --model.use_semantic false
done

echo "===== ALL DONE $(date +%H:%M:%S) ====="
python scripts/aggregate.py --runs ./runs --out ./runs/tables.tex
