"""Export the exact cold/warm split for external baselines.

Run from the project root (the folder that contains csarec/ and configs/):

  python scripts/export_split.py --config configs/default.yaml \
      --data.source amazon --data.category All_Beauty --output.tag csa_full

Writes TSV + JSON to  <output.dir>/<output.tag>/split/  (default: runs/<tag>/split/).
The split is deterministic given the same seed / cold_ratio / k-core settings,
so TALLRec, MMREC, GraphLoRA, etc. can train and evaluate on identical data.
"""
import os
import sys

# make the project root importable regardless of where python is invoked from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csarec.utils import load_config, set_seed          # noqa: E402
from csarec.data import load_data, export_split          # noqa: E402


def main():
    cfg = load_config()
    set_seed(cfg.seed)
    data = load_data(cfg)
    out_dir = os.path.join(cfg.dget("output.dir", "./runs"),
                           cfg.dget("output.tag", "run"), "split")
    meta = export_split(data, out_dir)
    print(f"[split] exported to {out_dir}")
    for k, v in meta.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
