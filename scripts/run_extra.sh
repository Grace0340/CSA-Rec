#!/usr/bin/env bash
# Supplementary experiment matrix that answers the reviewer-style gaps:
#   P1  learned cold-start baselines (Content2Emb / DropoutNet / CLCRec)
#   P2  seeds 2029-2030 for the significance set (n=5 for the key comparison)
#   P3  temporal (realistic) cold-start split
#   P4  encoder-scaling study (stronger frozen encoder: all-mpnet-base-v2, 768d)
#   P5  hyperparameter sensitivity (lambda / K / tau) on Video_Games
#   P6  offline near-duplicate / post-filter analysis
#
# Launch in the background and watch the log:
#   nohup bash scripts/run_extra.sh > run_extra.log 2>&1 &
#   tail -f run_extra.log
#
# Select a subset of phases:  PHASES="P1 P3" bash scripts/run_extra.sh
# Each run() skips a tag whose metrics.json already exists, so it is resumable.
set -u
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-~/autodl-tmp/hf_cache}

CFG=configs/default.yaml
CATS="Video_Games Baby_Products Toys_and_Games"
PHASES="${PHASES:-P1 P2 P3 P4 P5 P6}"
MPNET="sentence-transformers/all-mpnet-base-v2"

run() {  # run() <tag> <extra-args...>
  local tag=$1; shift
  if [ -f "runs/$tag/metrics.json" ]; then echo "[skip] $tag"; return; fi
  echo "===== $(date +%H:%M:%S) RUN $tag ====="
  python -m csarec.train --config $CFG --semantic.method sbert --device cuda \
      --data.source amazon "$@" --output.tag "$tag" || echo "[FAIL] $tag"
}

has() { echo " $PHASES " | grep -q " $1 "; }

for cat in $CATS; do
  python scripts/download_amazon.py --category "$cat" --root ./data || true
done

# ---------------------------------------------------------------- P1: learned baselines
if has P1; then
  echo "########## P1 learned cold-start baselines ##########"
  for cat in $CATS; do
    B="--data.category $cat"
    for seed in 2026 2027 2028; do
      run ${cat}_content2emb_s${seed} $B --seed $seed --method content2emb
      run ${cat}_dropoutnet_s${seed}  $B --seed $seed --method dropoutnet
      run ${cat}_clcrec_s${seed}      $B --seed $seed --method clcrec
    done
  done
fi

# ---------------------------------------------------------------- P2: extra seeds (n=5)
if has P2; then
  echo "########## P2 significance seeds 2029-2030 ##########"
  for cat in $CATS; do
    B="--data.category $cat"
    for seed in 2029 2030; do
      run ${cat}_full_s${seed}        $B --seed $seed
      run ${cat}_contentknn_s${seed}  $B --seed $seed --baseline content_knn
      run ${cat}_content2emb_s${seed} $B --seed $seed --method content2emb
      run ${cat}_dropoutnet_s${seed}  $B --seed $seed --method dropoutnet
      run ${cat}_clcrec_s${seed}      $B --seed $seed --method clcrec
    done
  done
fi

# ---------------------------------------------------------------- P3: temporal split
if has P3; then
  echo "########## P3 temporal cold-start split ##########"
  for cat in $CATS; do
    B="--data.category $cat --data.cold_split temporal"
    for seed in 2026 2027 2028; do
      run ${cat}_full_temporal_s${seed}       $B --seed $seed
      run ${cat}_ctrlcf_temporal_s${seed}     $B --seed $seed --model.use_semantic false
      run ${cat}_contentknn_temporal_s${seed} $B --seed $seed --baseline content_knn
    done
  done
fi

# ---------------------------------------------------------------- P4: encoder scaling
if has P4; then
  echo "########## P4 encoder-scaling study (all-mpnet-base-v2, 768d) ##########"
  for cat in $CATS; do
    B="--data.category $cat --semantic.model_name $MPNET"
    for seed in 2026 2027 2028; do
      run ${cat}_full_mpnet_s${seed}       $B --seed $seed
      run ${cat}_contentknn_mpnet_s${seed} $B --seed $seed --baseline content_knn
    done
  done
fi

# ---------------------------------------------------------------- P5: sensitivity (VG)
if has P5; then
  echo "########## P5 hyperparameter sensitivity (Video_Games, seed 2026) ##########"
  VG="--data.category Video_Games --seed 2026"
  for lam in 0.1 0.2 1.0; do run Video_Games_lam${lam}_s2026 $VG --model.csia_lambda $lam; done
  for K   in 5 20 40;      do run Video_Games_K${K}_s2026    $VG --model.cpg_neighbors $K; done
  for tau in 0.1 0.5;      do run Video_Games_tau${tau}_s2026 $VG --model.csia_tau $tau; done
fi

# ---------------------------------------------------------------- P6: near-duplicate analysis
if has P6; then
  echo "########## P6 near-duplicate / post-filter analysis ##########"
  for cat in $CATS; do
    PYTHONPATH=. python scripts/neardup_analysis.py --config $CFG --data.source amazon \
        --data.category $cat --semantic.method sbert --seed 2026 \
        --out runs/neardup_${cat}_s2026.json || echo "[FAIL] neardup $cat"
  done
fi

echo "===== EXTRA MATRIX DONE $(date +%H:%M:%S) ====="
python scripts/aggregate.py --runs ./runs --out ./runs/tables.tex || true
