#!/usr/bin/env python3
import json, csv, shutil, argparse
from pathlib import Path
from typing import List, Dict, Tuple
import random

def get_question_diversity_score(question: str) -> Dict[str, int]:
    """
    Analyze question characteristics for diversity selection.
    Returns a dictionary with various features for balancing.
    """
    q_lower = question.lower()

    # Question type indicators
    features = {
        'is_yes_no': 1 if any(word in q_lower for word in ['is there', 'are there', 'does', 'do']) else 0,
        'is_counting': 1 if any(word in q_lower for word in ['how many', 'what number']) else 0,
        'is_color': 1 if any(word in q_lower for word in ['color', 'red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'brown', 'gray']) else 0,
        'is_shape': 1 if any(word in q_lower for word in ['cube', 'sphere', 'cylinder', 'ball', 'block', 'shape']) else 0,
        'is_size': 1 if any(word in q_lower for word in ['large', 'small', 'big', 'tiny', 'size']) else 0,
        'is_material': 1 if any(word in q_lower for word in ['metal', 'rubber', 'matte', 'shiny', 'material']) else 0,
        'is_spatial': 1 if any(word in q_lower for word in ['left', 'right', 'behind', 'front', 'above', 'below']) else 0,
        'is_comparison': 1 if any(word in q_lower for word in ['same', 'different', 'more', 'less', 'equal']) else 0,
        'word_count': len(question.split()),
    }

    return features

def select_diverse_questions(questions: List[Dict], target_count: int = 200) -> List[Tuple[int, Dict]]:
    """
    Select diverse questions from CLEVR-Humans dataset to ensure variety.
    Returns list of (index, question) tuples.
    """

    print(f"Analyzing {len(questions)} CLEVR-Humans questions for diversity...")

    # Analyze all questions and group by characteristics
    question_features = []
    for idx, q in enumerate(questions):
        features = get_question_diversity_score(q['question'])
        question_features.append((idx, q, features))

    # Group questions by answer type for balanced sampling
    yes_no_questions = [(idx, q) for idx, q, f in question_features if q['answer'].lower() in ['yes', 'no']]
    counting_questions = [(idx, q) for idx, q, f in question_features if q['answer'].isdigit()]
    other_questions = [(idx, q) for idx, q, f in question_features if q['answer'].lower() not in ['yes', 'no'] and not q['answer'].isdigit()]

    print(f"Found {len(yes_no_questions)} yes/no, {len(counting_questions)} counting, {len(other_questions)} other questions")

    # Target distribution: balanced across types
    target_yes_no = min(80, len(yes_no_questions))  # ~40% yes/no
    target_counting = min(60, len(counting_questions))  # ~30% counting
    target_other = target_count - target_yes_no - target_counting  # remaining for other types

    # Randomly sample from each category for diversity
    random.seed(42)  # For reproducibility

    selected_yes_no = random.sample(yes_no_questions, min(target_yes_no, len(yes_no_questions)))
    selected_counting = random.sample(counting_questions, min(target_counting, len(counting_questions)))
    selected_other = random.sample(other_questions, min(target_other, len(other_questions)))

    # If we don't have enough of one type, fill with others
    all_selected = selected_yes_no + selected_counting + selected_other

    if len(all_selected) < target_count:
        remaining_questions = [(idx, q) for idx, q in enumerate(questions)
                             if (idx, q) not in all_selected]
        additional_needed = target_count - len(all_selected)
        additional = random.sample(remaining_questions,
                                 min(additional_needed, len(remaining_questions)))
        all_selected.extend(additional)

    # Shuffle for variety
    random.shuffle(all_selected)

    print(f"Selected {len(selected_yes_no)} yes/no, {len(selected_counting)} counting, {len(selected_other)} other questions")
    return all_selected[:target_count]

def export_subset(data_dir: str,
                  out_dir: str,
                  target_count: int = 200):
    """Export a diverse subset of CLEVR-Humans questions."""

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CLEVR-Humans questions
    humans_file = Path("CLEVR-Humans-train.json")
    if not humans_file.exists():
        raise FileNotFoundError(f"CLEVR-Humans-train.json not found in current directory")

    with humans_file.open('r') as f:
        humans_data = json.load(f)

    questions = humans_data["questions"]
    print(f"Loaded {len(questions)} CLEVR-Humans questions")

    # Select diverse questions
    selected_questions = select_diverse_questions(questions, target_count)

    # Setup image directories
    img_src_root = data_dir / "images" / "train"
    img_dst_root = out_dir / "images"
    img_dst_root.mkdir(exist_ok=True)

    rows = []
    for idx, q in selected_questions:
        print(f"Q{idx}: {q['question']} → {q['answer']}")

        # Copy corresponding image
        src = img_src_root / q["image_filename"]
        dst = img_dst_root / q["image_filename"]

        if src.exists():
            shutil.copy2(src, dst)
        else:
            print(f"Warning: Image {src} not found, skipping...")
            continue

        rows.append({
            "idx": idx,
            "image_index": q.get("image_index", idx),  # Include original image_index
            "image_filename": q["image_filename"],
            "question": q["question"],
            "answer": q["answer"],
            "difficulty": "human",  # Dummy difficulty value as requested
            "split": q.get("split", "train"),  # Include original split info
            "image_path": str(dst.relative_to(out_dir)),
        })

    # Export to CSV
    csv_path = out_dir / "subset.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Print summary statistics
    yes_no_count = sum(1 for row in rows if str(row['answer']).lower() in ['yes', 'no'])
    counting_count = sum(1 for row in rows if str(row['answer']).isdigit())
    other_count = len(rows) - yes_no_count - counting_count

    # Analyze question characteristics
    color_questions = sum(1 for row in rows if any(color in row['question'].lower()
                         for color in ['red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'brown', 'gray', 'color']))
    shape_questions = sum(1 for row in rows if any(shape in row['question'].lower()
                         for shape in ['cube', 'sphere', 'cylinder', 'ball', 'block', 'shape']))
    size_questions = sum(1 for row in rows if any(size in row['question'].lower()
                        for size in ['large', 'small', 'big', 'tiny', 'size']))

    print(f"\n✔ {len(rows)} CLEVR-Humans questions exported to {out_dir}")
    print(f"   Answer Distribution:")
    print(f"    • Yes/No questions: {yes_no_count}")
    print(f"    • Counting questions: {counting_count}")
    print(f"    • Other questions: {other_count}")
    print(f"   Question Topics:")
    print(f"    • Color-related: {color_questions}")
    print(f"    • Shape-related: {shape_questions}")
    print(f"    • Size-related: {size_questions}")
    print(f"   Files:")
    print(f"    • CSV: {csv_path.name}")
    print(f"    • Images: {img_dst_root.relative_to(out_dir)}/*")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export a diverse subset of CLEVR-Humans questions")
    ap.add_argument("--data-dir", default="clevr",
                    help="CLEVR_v1.0 root (contains images/ directory)")
    ap.add_argument("--out-dir", default="clevr_human_subset",
                    help="Destination folder for the human subset")
    ap.add_argument("--count", type=int, default=200,
                    help="Number of questions to export (default: 200)")
    args = ap.parse_args()

    export_subset(args.data_dir, args.out_dir, args.count)
