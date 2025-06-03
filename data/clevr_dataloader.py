#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import Dict, List

class CLEVRDataloader:
    def __init__(self, data_dir: str = "clevr"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images" / "train"
        self.questions_file = self.data_dir / "questions" / "CLEVR_train_questions.json"

        with open(self.questions_file, 'r') as f:
            data = json.load(f)
            self.questions = data['questions']

    def get_sample(self, idx: int) -> Dict:
        question_data = self.questions[idx]

        sample = {
            'image_filename': question_data['image_filename'],
            'image_path': str(self.images_dir / question_data['image_filename']),
            'question': question_data['question'],
            'answer': question_data['answer'],
            'question_family_index': question_data.get('question_family_index', -1),
            'program': question_data.get('program', [])
        }
        return sample

    def __len__(self):
        return len(self.questions)

if __name__ == "__main__":
    dataloader = CLEVRDataloader()

    sample = dataloader.get_sample(0)
    print(f"Image: {sample['image_filename']}")
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
    print(f"Image path: {sample['image_path']}")
