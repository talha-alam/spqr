#!/usr/bin/env python
"""
Benign fine-tuning (BFT) dataset preparation utility.

The SPQR BFT trainer (`spqr/attacks/bft_trainer.py`) consumes datasets in the
HuggingFace `imagefolder` layout: a directory of images plus a `metadata.jsonl`
file mapping each `file_name` to its caption `text`. This script builds that
layout for the benign scenarios used in the paper:

  * general        — COCO subset (image/caption pairs)
  * multilingual   — COCO captions translated to ar/es/fr/hi
  * domain         — artistic / medical image+caption folders

Subcommands
-----------
  coco       Build an imagefolder from COCO-style annotations (captions JSON).
  from-csv   Build an imagefolder from a CSV/JSON of (image_path, caption).
  verify     Validate an existing imagefolder layout.

Examples
--------
  # 1) General scenario: 5,000 COCO pairs
  python scripts/prepare_datasets.py coco \
      --images_dir /data/coco/train2017 \
      --captions_json /data/coco/annotations/captions_train2017.json \
      --output_dir data/bft_datasets/coco --num_samples 5000 --seed 42

  # 2) Domain/multilingual: from a (image_path, caption) table
  python scripts/prepare_datasets.py from-csv \
      --table data/raw/artistic.csv \
      --output_dir data/bft_datasets/artistic --num_samples 5000

  # 3) Sanity-check a prepared folder
  python scripts/prepare_datasets.py verify --output_dir data/bft_datasets/coco
"""
import argparse
import csv
import json
import os
import random
import shutil


def _write_metadata(output_dir, records):
    """Write a metadata.jsonl with {"file_name", "text"} entries."""
    meta_path = os.path.join(output_dir, "metadata.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({"file_name": rec["file_name"], "text": rec["text"]},
                               ensure_ascii=False) + "\n")
    return meta_path


def _copy_image(src, dst_dir):
    """Copy an image into dst_dir, returning its basename (or None on failure)."""
    if not os.path.isfile(src):
        return None
    fname = os.path.basename(src)
    shutil.copy2(src, os.path.join(dst_dir, fname))
    return fname


def prepare_coco(args):
    with open(args.captions_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Map image_id -> file_name and collect one caption per image.
    id_to_file = {img["id"]: img["file_name"] for img in coco.get("images", [])}
    caption_by_id = {}
    for ann in coco.get("annotations", []):
        caption_by_id.setdefault(ann["image_id"], ann["caption"].strip())

    pairs = [(id_to_file[i], cap) for i, cap in caption_by_id.items() if i in id_to_file]
    if not pairs:
        raise SystemExit("No (image, caption) pairs found in the annotations file.")

    random.seed(args.seed)
    random.shuffle(pairs)

    os.makedirs(args.output_dir, exist_ok=True)
    records, n = [], 0
    for file_name, caption in pairs:
        if n >= args.num_samples:
            break
        copied = _copy_image(os.path.join(args.images_dir, file_name), args.output_dir)
        if copied is None:
            continue
        records.append({"file_name": copied, "text": caption})
        n += 1

    _write_metadata(args.output_dir, records)
    print(f"Prepared {len(records)} COCO pairs -> {args.output_dir}")


def _read_table(path):
    """Read a CSV or JSON table of {image_path, caption} rows."""
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        return [(r.get("image_path") or r.get("image"), r.get("caption") or r.get("text"))
                for r in rows]
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            img = r.get("image_path") or r.get("image") or r.get("file_name")
            cap = r.get("caption") or r.get("text") or ""
            rows.append((img, cap))
    return rows


def prepare_from_csv(args):
    rows = [(img, cap) for img, cap in _read_table(args.table) if img]
    if not rows:
        raise SystemExit("No usable rows found in the table.")

    random.seed(args.seed)
    random.shuffle(rows)

    os.makedirs(args.output_dir, exist_ok=True)
    records, n = [], 0
    for img_path, caption in rows:
        if n >= args.num_samples:
            break
        src = img_path if os.path.isabs(img_path) else os.path.join(args.images_root, img_path)
        copied = _copy_image(src, args.output_dir)
        if copied is None:
            continue
        records.append({"file_name": copied, "text": (caption or "").strip()})
        n += 1

    _write_metadata(args.output_dir, records)
    print(f"Prepared {len(records)} pairs -> {args.output_dir}")


def verify(args):
    meta = os.path.join(args.output_dir, "metadata.jsonl")
    if not os.path.isfile(meta):
        raise SystemExit(f"Missing metadata.jsonl in {args.output_dir}")

    n_ok, n_missing, n_empty = 0, 0, 0
    with open(meta, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            img = os.path.join(args.output_dir, rec["file_name"])
            if not os.path.isfile(img):
                n_missing += 1
                continue
            if not rec.get("text"):
                n_empty += 1
            n_ok += 1

    print(f"Verified {args.output_dir}: {n_ok} valid entries, "
          f"{n_missing} missing images, {n_empty} empty captions.")
    if n_missing:
        raise SystemExit("Some referenced images are missing — fix before training.")


def main():
    parser = argparse.ArgumentParser(description="Prepare benign BFT datasets (imagefolder layout).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_coco = sub.add_parser("coco", help="Build an imagefolder from COCO captions.")
    p_coco.add_argument("--images_dir", required=True)
    p_coco.add_argument("--captions_json", required=True)
    p_coco.add_argument("--output_dir", required=True)
    p_coco.add_argument("--num_samples", type=int, default=5000)
    p_coco.add_argument("--seed", type=int, default=42)
    p_coco.set_defaults(func=prepare_coco)

    p_csv = sub.add_parser("from-csv", help="Build an imagefolder from a CSV/JSON table.")
    p_csv.add_argument("--table", required=True, help="CSV/JSON with image_path + caption columns.")
    p_csv.add_argument("--images_root", default="", help="Root prepended to relative image paths.")
    p_csv.add_argument("--output_dir", required=True)
    p_csv.add_argument("--num_samples", type=int, default=5000)
    p_csv.add_argument("--seed", type=int, default=42)
    p_csv.set_defaults(func=prepare_from_csv)

    p_ver = sub.add_parser("verify", help="Validate an existing imagefolder layout.")
    p_ver.add_argument("--output_dir", required=True)
    p_ver.set_defaults(func=verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
