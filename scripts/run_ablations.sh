#!/usr/bin/env bash
# Reproduce the ablation study (Section V-D). Adjust --data.* for your dataset.
set -e
CFG=configs/default.yaml
COMMON="--config $CFG --data.source amazon --data.category Beauty --semantic.method sbert"

# Full CSA-Rec
python -m csarec.train $COMMON --output.tag csa_full

# w/o CSIA alignment loss
python -m csarec.train $COMMON --model.use_csia false --output.tag abl_no_csia

# w/o CPG (mean-warm fallback for cold items)
python -m csarec.train $COMMON --model.use_cpg false --output.tag abl_no_cpg

# w/o SAA low-rank (full-rank linear adapter)
python -m csarec.train $COMMON --model.saa_type linear --output.tag abl_saa_linear

# Pure CF (no semantics -> no cold-start capability): lower bound
python -m csarec.train $COMMON --model.use_semantic false --output.tag abl_pure_cf

# Backbone swap
python -m csarec.train $COMMON --model.backbone mf --output.tag csa_mf

# Adapter rank sensitivity
for r in 8 16 32 64; do
  python -m csarec.train $COMMON --model.saa_rank $r --output.tag "rank_$r"
done
