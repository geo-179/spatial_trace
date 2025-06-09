#!/usr/bin/env python3
import json, csv, shutil, argparse
from pathlib import Path
from typing import Sequence

EASY_QUESTIONS_IDX = [
     1422, 2372, 4112, 5152, 8732, 10243, 13103, 20453, 27033, 31553
]

def export_subset(data_dir: str,
                  out_dir: str,
                  idx_list: Sequence[int] = EASY_QUESTIONS_IDX):
    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qfile = data_dir / "questions" / "CLEVR_train_questions.json"
    questions = json.loads(qfile.read_text())["questions"]

    img_src_root = data_dir / "images" / "train"
    img_dst_root = out_dir / "images"
    img_dst_root.mkdir(exist_ok=True)

    rows = []
    for idx in idx_list:
        q = questions[idx]
        print(q)
        src = img_src_root / q["image_filename"]
        dst = img_dst_root / q["image_filename"]
        shutil.copy2(src, dst)

        rows.append({
            "idx": idx,
            "image_filename": q["image_filename"],
            "question": q["question"],
            "answer": q["answer"],
            "image_path": str(dst.relative_to(out_dir)),
        })

    csv_path = out_dir / "subset.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {len(rows)} samples exported to {out_dir}")
    print(f"   CSV:    {csv_path.name}")
    print(f"   Images: {img_dst_root.relative_to(out_dir)}/*")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="clevr",
                    help="CLEVR_v1.0 root (contains images/ and questions/)")
    ap.add_argument("--out-dir",  default="clevr_easy_subset",
                    help="Destination folder for the subset")
    args = ap.parse_args()
    export_subset(args.data_dir, args.out_dir)
