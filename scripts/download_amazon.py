"""Download Amazon Reviews 2023 (McAuley Lab) category files.

Usage:
  python scripts/download_amazon.py --category Beauty --root ./data

Files are placed under <root>/amazon/<Category>/ as expected by csarec.data.load_amazon.
Public dataset homepage: https://amazon-reviews-2023.github.io/
"""
import os
import argparse
import urllib.request

BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"


def _download(url, dst):
    if os.path.exists(dst):
        print(f"[skip] {dst} already exists")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"[get ] {url}")
    urllib.request.urlretrieve(url, dst)
    print(f"[ok  ] {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Beauty",
                    help="e.g. All_Beauty, Baby_Products, Sports_and_Outdoors, Toys_and_Games")
    ap.add_argument("--root", default="./data")
    args = ap.parse_args()

    out = os.path.join(args.root, "amazon", args.category)
    _download(f"{BASE}/review_categories/{args.category}.jsonl.gz",
              os.path.join(out, f"{args.category}.jsonl.gz"))
    _download(f"{BASE}/meta_categories/meta_{args.category}.jsonl.gz",
              os.path.join(out, f"meta_{args.category}.jsonl.gz"))
    print("\nDone. Now run:")
    print(f"  python -m csarec.train --config configs/default.yaml "
          f"--data.source amazon --data.category {args.category} --semantic.method sbert")


if __name__ == "__main__":
    main()
