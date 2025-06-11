#!/usr/bin/env python3
import json, csv, shutil, argparse, re
from pathlib import Path
from typing import Sequence, List, Dict, Tuple
import random

def classify_question_difficulty(question: str, answer: str) -> str:
    """
    Classify a question as 'easy' or 'hard' based on linguistic complexity.

    Easy questions (1 tool call):
    - Simple counting: "How many X are there?"
    - Direct property queries: "What color is X?"
    - Simple existence: "Is there a X?"

    Hard questions (multiple tool calls):
    - Comparisons: "same size as", "same color as", "same shape as"
    - Conditional logic: "behind the X", "in front of the Y"
    - Multi-step reasoning: "the thing that is left of the thing that is..."
    """

    # Convert to lowercase for analysis
    q_lower = question.lower()

    # Hard question indicators
    hard_indicators = [
        'same size as', 'same color as', 'same shape as', 'same material as',
        'behind the', 'in front of', 'left of', 'right of',
        'that is behind', 'that is in front', 'that is left', 'that is right',
        'there is a', 'there is an', 'there are',
        'less than', 'greater than', 'more than',
        'the thing that', 'the object that',
        'does it have the same', 'is it the same',
        'number of.*less than', 'number of.*greater than'
    ]

    # Easy question indicators
    easy_indicators = [
        'how many', 'what number of', 'what color', 'what shape', 'what material',
        'is there a', 'are there any',
        'what is the color', 'what is the shape', 'what is the material'
    ]

    # Count hard vs easy indicators
    hard_score = sum(1 for indicator in hard_indicators if indicator in q_lower)
    easy_score = sum(1 for indicator in easy_indicators if indicator in q_lower)

    # Additional complexity factors
    complexity_factors = len(re.findall(r'\bthat\b', q_lower))  # "that" indicates nested references
    word_count = len(question.split())

    # Decision logic
    if hard_score > easy_score or complexity_factors >= 2 or word_count > 15:
        return 'hard'
    else:
        return 'easy'

def is_yes_no_or_counting(answer: str) -> bool:
    """Check if answer is yes/no or a counting number."""
    answer_lower = str(answer).lower().strip()
    return answer_lower in ['yes', 'no'] or answer_lower.isdigit()

def select_mixed_questions(questions: List[Dict], target_count: int = 100) -> List[Tuple[int, Dict]]:
    """
    Select a balanced mix of easy and hard yes/no and counting questions.
    Returns list of (index, question) tuples.
    """

    # Filter for yes/no and counting questions
    valid_questions = []
    for idx, q in enumerate(questions):
        if is_yes_no_or_counting(q['answer']):
            difficulty = classify_question_difficulty(q['question'], q['answer'])
            valid_questions.append((idx, q, difficulty))

    print(f"Found {len(valid_questions)} valid yes/no and counting questions")

    # Separate easy and hard questions
    easy_questions = [(idx, q) for idx, q, diff in valid_questions if diff == 'easy']
    hard_questions = [(idx, q) for idx, q, diff in valid_questions if diff == 'hard']

    print(f"Easy questions: {len(easy_questions)}")
    print(f"Hard questions: {len(hard_questions)}")

    # Target distribution: 60% easy, 40% hard for balanced learning
    target_easy = int(target_count * 0.6)  # 60 easy questions
    target_hard = target_count - target_easy  # 40 hard questions

    # Randomly sample from each category
    random.seed(42)  # For reproducibility
    selected_easy = random.sample(easy_questions, min(target_easy, len(easy_questions)))
    selected_hard = random.sample(hard_questions, min(target_hard, len(hard_questions)))

    # If we don't have enough of one type, fill with the other
    total_selected = len(selected_easy) + len(selected_hard)
    if total_selected < target_count:
        remaining = target_count - total_selected
        if len(selected_easy) < target_easy:
            # Need more easy questions
            remaining_easy = [q for q in easy_questions if q not in selected_easy]
            selected_easy.extend(random.sample(remaining_easy, min(remaining, len(remaining_easy))))
        else:
            # Need more hard questions
            remaining_hard = [q for q in hard_questions if q not in selected_hard]
            selected_hard.extend(random.sample(remaining_hard, min(remaining, len(remaining_hard))))

    # Combine and shuffle
    all_selected = selected_easy + selected_hard
    random.shuffle(all_selected)

    print(f"Selected {len(selected_easy)} easy and {len(selected_hard)} hard questions")
    return all_selected[:target_count]

def export_subset(data_dir: str,
                  out_dir: str,
                  target_count: int = 100):
    """Export a mixed subset of CLEVR questions."""

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qfile = data_dir / "questions" / "CLEVR_train_questions.json"
    questions = json.loads(qfile.read_text())["questions"]

    print(f"Loaded {len(questions)} total questions from CLEVR dataset")

    # Select mixed questions
    selected_questions = select_mixed_questions(questions, target_count)

    img_src_root = data_dir / "images" / "train"
    img_dst_root = out_dir / "images"
    img_dst_root.mkdir(exist_ok=True)

    rows = []
    for idx, q in selected_questions:
        print(f"Q{idx}: {q['question']} → {q['answer']} ({classify_question_difficulty(q['question'], q['answer'])})")

        src = img_src_root / q["image_filename"]
        dst = img_dst_root / q["image_filename"]
        shutil.copy2(src, dst)

        rows.append({
            "idx": idx,
            "image_filename": q["image_filename"],
            "question": q["question"],
            "answer": q["answer"],
            "difficulty": classify_question_difficulty(q["question"], q["answer"]),
            "image_path": str(dst.relative_to(out_dir)),
        })

    csv_path = out_dir / "subset.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Print summary statistics
    easy_count = sum(1 for row in rows if row['difficulty'] == 'easy')
    hard_count = sum(1 for row in rows if row['difficulty'] == 'hard')
    yes_no_count = sum(1 for row in rows if str(row['answer']).lower() in ['yes', 'no'])
    counting_count = sum(1 for row in rows if str(row['answer']).isdigit())

    print(f"\n✔ {len(rows)} mixed questions exported to {out_dir}")
    print(f"   Distribution:")
    print(f"    • Easy questions: {easy_count}")
    print(f"    • Hard questions: {hard_count}")
    print(f"    • Yes/No questions: {yes_no_count}")
    print(f"    • Counting questions: {counting_count}")
    print(f"   Files:")
    print(f"    • CSV: {csv_path.name}")
    print(f"    • Images: {img_dst_root.relative_to(out_dir)}/*")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export a mixed subset of CLEVR questions (easy + hard)")
    ap.add_argument("--data-dir", default="clevr",
                    help="CLEVR_v1.0 root (contains images/ and questions/)")
    ap.add_argument("--out-dir", default="clevr_mixed_subset",
                    help="Destination folder for the mixed subset")
    ap.add_argument("--count", type=int, default=100,
                    help="Number of questions to export (default: 100)")
    args = ap.parse_args()

    export_subset(args.data_dir, args.out_dir, args.count)
