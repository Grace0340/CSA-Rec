# CSA-Rec

Lightweight cold-start recommendation on top of a frozen text encoder and a
standard collaborative filtering backbone.

New items in a catalog have no interaction history, so ID-based collaborative
models cannot rank them. CSA-Rec keeps the language model frozen (used once,
offline, as a feature extractor) and keeps the collaborative backbone
(LightGCN or MF) in its usual role, then trains a small bridge between the two:

- **SAA** — a low-rank adapter that projects frozen sentence embeddings into
  the collaborative embedding space (~7K parameters).
- **CSIA** — a bidirectional InfoNCE loss that aligns the adapted semantics
  with the ID embeddings of warm items. A stop-gradient on the ID side keeps
  the alignment from disturbing the collaborative signal.
- **CPG** — cold items are represented by an attention-weighted aggregation of
  the *trained* ID embeddings of their nearest semantic warm neighbors,
  refined by a small linear map.
- **Adaptive fusion** — a single learnable scalar mixes the collaborative
  score with a direct content-match score. It calibrates itself to the
  sparsity of the dataset; no per-dataset tuning.

The whole trainable bridge is ~11K parameters regardless of catalog size.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# for GPU, install the CUDA build of torch: https://pytorch.org/get-started/locally/
```

## Quick start (no data, no GPU needed)

A synthetic smoke test validates the full pipeline in a minute or two on CPU:

```bash
bash scripts/smoke_test.sh          # Linux/Mac
./scripts/smoke_test.ps1            # Windows PowerShell
```

Outputs land in `runs/smoke/`. Expect non-trivial `cold` metrics — the
synthetic generator injects a semantic signal correlated with the latent item
factors.

## Real data

Experiments use the public **Amazon Reviews 2023** dataset released by the
McAuley Lab (UCSD): <https://amazon-reviews-2023.github.io/>. If you use it,
please cite the dataset per the instructions on that page. The download helper
fetches per-category review and metadata files:

```bash
python scripts/download_amazon.py --category Video_Games --root ./data

python -m csarec.train --config configs/default.yaml \
    --data.source amazon --data.category Video_Games \
    --semantic.method sbert --device cuda
```

Text encoding defaults to `sentence-transformers/all-MiniLM-L6-v2`, executed
once per catalog and cached under `data/`. CLIP (text+image) is available with
`--semantic.method clip --semantic.use_image true`.

## Cold-start protocol

- Ratings >= 4 are implicit positives, followed by iterative 5-core filtering.
- A fraction `data.cold_ratio` (default 10%) of items is held out entirely:
  all of their interactions are removed from training.
- Warm items use per-user leave-one-out (last -> test, second-to-last -> valid).
- Evaluation reports warm / cold / all separately, by full ranking over the
  respective candidate pool with train positives masked.

`scripts/export_split.py` dumps the exact split (user/item maps, train/valid/
test TSVs) so external methods can be evaluated on identical partitions.

## Reproducing the experiment matrix

```bash
nohup bash scripts/run_all.sh > run_all.log 2>&1 &
```

runs three categories x three seeds x {full, w/o CSIA, w/o CPG, w/o fusion,
CF-only, content-kNN, popularity, random} plus backbone/rank ablations and a
cold-ratio severity sweep. Aggregate results afterwards:

```bash
python scripts/aggregate.py --runs ./runs --out ./runs/tables.tex
python scripts/make_paper_assets.py --runs ./runs --out ./paper_assets
```

The first prints per-run and mean±std tables; the second renders the summary
figures and LaTeX tables from `runs/*/metrics.json`.

### Ablation flags

| Flag | Meaning |
|------|---------|
| `--model.use_csia false` | drop the contrastive alignment loss |
| `--model.use_cpg false` | cold items use the adapter output only |
| `--model.use_fusion false` | remove the content-match score term |
| `--model.use_semantic false` | pure CF (cold-start lower bound) |
| `--model.backbone mf` | MF instead of LightGCN |
| `--model.saa_type linear` | full-rank linear adapter |
| `--model.saa_rank {8,16,32,64}` | bottleneck size |
| `--baseline {content_knn,pop,random}` | training-free reference scorers |

## Outputs

Each run writes `runs/<tag>/metrics.json`:

```json
{ "metrics": { "warm": {...}, "cold": {...}, "all": {...} },
  "efficiency": { "trainable_params": ..., "adapter_params": ...,
                  "adapter_ratio": ..., "full_inference_ms": ...,
                  "fusion_beta": ... } }
```

plus `rationales.txt` with human-readable explanations of sample cold-item
recommendations derived from the CPG attention weights.

## Code map

| File | Contents |
|------|----------|
| `csarec/data.py` | loaders, k-core filtering, cold split, split export |
| `csarec/semantic.py` | frozen encoders (SBERT / CLIP / hash fallback) + cache |
| `csarec/models.py` | LightGCN/MF backbone, SAA, CSIA, CPG, fusion |
| `csarec/baselines.py` | content-kNN / popularity / random scorers |
| `csarec/metrics.py` | Recall@K / NDCG@K with candidate-pool restriction |
| `csarec/train.py` | training loop, evaluation, efficiency measurement |

## Practical notes

- Adam lr 1e-2 with zero weight decay is deliberate: at these data scales a
  1e-3 rate lets the BPR gradient collapse the embeddings to the zero fixed
  point, and global weight decay does the same more slowly.
- Full-ranking evaluation loops per user for clarity; for very large catalogs
  switch to sampled metrics or batch the masking.
- CLIP image encoding requires downloading item images (URLs are in the
  metadata); text-only SBERT is the reliable default.

## License

MIT — see [LICENSE](LICENSE).
