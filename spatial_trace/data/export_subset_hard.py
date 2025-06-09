#!/usr/bin/env python3
import json, csv, shutil
from pathlib import Path
from typing import Sequence

HARD_QUESTIONS_IDX = [1421, 2375, 4113, 5156, 8732,
                      10246, 13104, 20459, 27035, 31551]

def export_subset(data_dir: str,
                  out_dir: str,
                  idx_list: Sequence[int] = HARD_QUESTIONS_IDX):
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
        src  = img_src_root / q["image_filename"]
        dst  = img_dst_root / q["image_filename"]
        shutil.copy2(src, dst)          # preserves timestamp/metadata

        rows.append({
            "idx"            : idx,
            "image_filename" : q["image_filename"],
            "question"       : q["question"],
            "answer"         : q["answer"],
            "image_path"     : str(dst.relative_to(out_dir))  # nice & short
        })

    csv_path = out_dir / "subset.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"✔  {len(rows)} samples exported to {out_dir}")
    print(f"   – CSV:    {csv_path.name}")
    print(f"   – Images: {img_dst_root.relative_to(out_dir)}/*")

if __name__ == "__main__":
    export_subset(data_dir="clevr",
                  out_dir="clevr_hard_subset")
