"""
Utilities for evaluation scripts.
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def load_traces(traces_path: Path) -> List[Dict[str, Any]]:
    """Load reasoning traces from JSON file."""
    try:
        with open(traces_path, 'r') as f:
            traces = json.load(f)

        # Handle both single trace and list of traces
        if isinstance(traces, dict):
            return [traces]
        return traces
    except Exception as e:
        logger.error(f"Error loading traces from {traces_path}: {e}")
        return []


def load_ground_truth_csv(csv_path: Path,
                         question_col: str = "question",
                         answer_col: str = "answer") -> Dict[str, str]:
    """Load ground truth from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        return dict(zip(df[question_col], df[answer_col]))
    except Exception as e:
        logger.error(f"Error loading ground truth from {csv_path}: {e}")
        return {}


def load_ground_truth_json(json_path: Path) -> Dict[str, str]:
    """Load ground truth from JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading ground truth from {json_path}: {e}")
        return {}


def save_results(results: Any, output_path: Path):
    """Save results to JSON file."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results to {output_path}: {e}")


def calculate_exact_match_accuracy(predictions: List[str],
                                 ground_truth: List[str]) -> float:
    """Calculate exact match accuracy."""
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    if not predictions:
        return 0.0

    correct = sum(
        pred.strip().lower() == gt.strip().lower()
        for pred, gt in zip(predictions, ground_truth)
    )
    return correct / len(predictions)


def extract_final_answers(traces: List[Dict[str, Any]]) -> List[Optional[str]]:
    """Extract final answers from a list of traces."""
    answers = []

    for trace_data in traces:
        trace = trace_data.get("trace", [])
        answer = None

        # Look for final answer in reverse order
        for message in reversed(trace):
            if message.get("role") == "assistant":
                try:
                    content = json.loads(message.get("content", "{}"))
                    if content.get("action") == "final_answer":
                        answer = content.get("text", "").strip()
                        break
                except json.JSONDecodeError:
                    continue

        answers.append(answer)

    return answers


def create_evaluation_summary(results: Dict[str, Any]) -> str:
    """Create a human-readable summary of evaluation results."""
    summary = "Evaluation Summary\n"
    summary += "=" * 30 + "\n"

    for key, value in results.items():
        if isinstance(value, float):
            summary += f"{key}: {value:.3f}\n"
        elif isinstance(value, (int, str)):
            summary += f"{key}: {value}\n"
        elif isinstance(value, dict):
            summary += f"{key}:\n"
            for subkey, subvalue in value.items():
                if isinstance(subvalue, float):
                    summary += f"  {subkey}: {subvalue:.3f}\n"
                else:
                    summary += f"  {subkey}: {subvalue}\n"

    return summary


def setup_evaluation_logging(log_level: str = "INFO"):
    """Setup logging for evaluation scripts."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('evaluation/evaluation.log')
        ]
    )


if __name__ == "__main__":
    # Test utilities
    setup_evaluation_logging()

    # Test loading traces if they exist
    if Path("evaluation/quality_traces.json").exists():
        traces = load_traces(Path("evaluation/quality_traces.json"))
        print(f"Loaded {len(traces)} traces")

        answers = extract_final_answers(traces)
        print(f"Extracted {len([a for a in answers if a])} final answers")
