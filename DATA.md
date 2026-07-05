# Data

CSA-Rec is evaluated on the public **Amazon Reviews 2023** corpus released by the
McAuley Lab (UCSD).

- Project page: https://amazon-reviews-2023.github.io/
- Categories used in the paper: `Video_Games`, `Baby_Products`, `Toys_and_Games`
- If you use this data, please cite it per the instructions on the project page.

## What is (and isn't) in this repository

The raw per-category dumps are large (hundreds of MB to >1 GB each) and exceed
GitHub's 100 MB per-file limit, so they are **not** committed here. They are
reproducible with the download helper:

```bash
python scripts/download_amazon.py --category Video_Games   --root ./data
python scripts/download_amazon.py --category Baby_Products --root ./data
python scripts/download_amazon.py --category Toys_and_Games --root ./data
```

This fetches, under `data/amazon/<Category>/`:

- `<Category>.jsonl.gz` — user–item reviews (implicit feedback)
- `meta_<Category>.jsonl.gz` — item metadata (title, features, description)

## Derived artifacts included

To make the pipeline runnable without re-encoding, the following cached,
regenerable artifacts are provided under `data/` (all well under 100 MB):

- `amazon_sbert_sem.npy`, `amazon_hash_sem.npy` — cached frozen item embeddings
  (all-MiniLM-L6-v2 / hash fallback).
- `amazon_<Category>_nbr_K10_cr*_seed*.npz` — precomputed warm-neighbor tables
  (K=10) used by the CPG module, per category / cold-ratio / seed.

All of these are rebuilt automatically from the raw data on first run if absent.

## Preprocessing (summary)

- Ratings >= 4 become implicit positives; iterative 5-core filtering on users
  and items.
- Item text = title + feature bullets + description, truncated to 2,000 chars.
- Cold split: a fraction `data.cold_ratio` (default 10%) of items has *all* its
  interactions removed from training (strict zero-interaction protocol);
  `data.cold_split: temporal` instead holds out the newest items.
- Warm items use per-user leave-one-out (last -> test, second-to-last -> valid).

See `configs/default.yaml` and `csarec/data.py` for the exact parameters, and
`scripts/export_split.py` to dump the exact per-seed partitions.
