"""
Test accuracy under different conditions for ablation study.
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
import logging

from spatial_trace import SpatialReasoningPipeline
from spatial_trace.utils import read_csv_data

logger = logging.getLogger(__name__)


@dataclass
class AccuracyResult:
    """Results for one test condition."""
    condition_name: str
    accuracy: float
    total_samples: int
    correct_answers: int
    failed_traces: int
    details: List[Dict[str, Any]]


class AccuracyTester:
    """Test system accuracy under different conditions."""

    def __init__(self):
        self.test_conditions = {
            "baseline": {
                "enable_verification": True,
                "min_acceptable_rating": 6.0,
                "max_steps": 10
            },
            "no_verification": {
                "enable_verification": False,
                "max_steps": 10
            },
            "high_standards": {
                "enable_verification": True,
                "min_acceptable_rating": 8.0,
                "max_steps": 10
            },
            "limited_steps": {
                "enable_verification": True,
                "min_acceptable_rating": 6.0,
                "max_steps": 5
            }
        }

    def test_condition(self,
                      condition_name: str,
                      dataset_path: Path,
                      ground_truth_path: Path = None,
                      max_samples: int = 20) -> AccuracyResult:
        """
        Test accuracy under one condition.

        Args:
            condition_name: Which test condition to run
            dataset_path: Test dataset CSV
            ground_truth_path: Ground truth answers (optional)
            max_samples: Number of samples to test

        Returns:
            Accuracy results for this condition
        """
        if condition_name not in self.test_conditions:
            raise ValueError(f"Unknown condition: {condition_name}")

        config = self.test_conditions[condition_name]
        logger.info(f"Testing condition '{condition_name}' with config: {config}")

        # Setup pipeline for this condition
        pipeline = SpatialReasoningPipeline(**config)

        # Load dataset
        data = read_csv_data(dataset_path)
        if data is None:
            raise ValueError(f"Could not load dataset from {dataset_path}")

        # Load ground truth if available
        ground_truth = {}
        if ground_truth_path and ground_truth_path.exists():
            if ground_truth_path.suffix == '.json':
                with open(ground_truth_path, 'r') as f:
                    ground_truth = json.load(f)
            else:
                # Assume CSV with columns: image_path, answer
                gt_data = pd.read_csv(ground_truth_path)
                ground_truth = dict(zip(gt_data['image_path'], gt_data['answer']))

        results = []
        correct_count = 0
        failed_count = 0

        # Test samples
        for idx, row in data.head(max_samples).iterrows():
            try:
                question = row['question']
                image_path = Path(dataset_path).parent / row['image_path']

                logger.info(f"Testing sample {idx}: {question}")

                # Generate trace
                trace = pipeline.generate_reasoning_trace(question, image_path)

                if trace:
                    # Extract final answer
                    final_answer = self._extract_final_answer(trace)

                    # Check correctness
                    is_correct = False
                    expected_answer = ground_truth.get(row['image_path']) or ground_truth.get(str(image_path))

                    if expected_answer and final_answer:
                        is_correct = self._check_answer_match(final_answer, expected_answer)
                        if is_correct:
                            correct_count += 1

                    results.append({
                        "sample_id": idx,
                        "question": question,
                        "predicted_answer": final_answer,
                        "expected_answer": expected_answer,
                        "is_correct": is_correct,
                        "trace_length": len(trace)
                    })
                else:
                    failed_count += 1
                    logger.error(f"Failed to generate trace for sample {idx}")
                    results.append({
                        "sample_id": idx,
                        "question": question,
                        "predicted_answer": None,
                        "expected_answer": expected_answer,
                        "is_correct": False,
                        "trace_length": 0
                    })

            except Exception as e:
                failed_count += 1
                logger.error(f"Error processing sample {idx}: {e}")

        # Calculate accuracy
        total_tested = len(results)
        accuracy = correct_count / total_tested if total_tested > 0 else 0.0

        return AccuracyResult(
            condition_name=condition_name,
            accuracy=accuracy,
            total_samples=total_tested,
            correct_answers=correct_count,
            failed_traces=failed_count,
            details=results
        )

    def _extract_final_answer(self, trace: List[Dict[str, Any]]) -> str:
        """Extract final answer from trace."""
        for message in reversed(trace):
            if message.get("role") == "assistant":
                try:
                    content = json.loads(message.get("content", "{}"))
                    if content.get("action") == "final_answer":
                        return content.get("text", "").strip()
                except json.JSONDecodeError:
                    continue
        return ""

    def _check_answer_match(self, predicted: str, expected: str) -> bool:
        """Check if predicted answer matches expected."""
        pred_clean = predicted.lower().strip()
        exp_clean = expected.lower().strip()

        # Exact match
        if pred_clean == exp_clean:
            return True

        # For yes/no questions, check if answer starts with expected
        if exp_clean in ["yes", "no"]:
            return pred_clean.startswith(exp_clean)

        # For numeric answers, try to extract numbers
        try:
            pred_num = float(pred_clean)
            exp_num = float(exp_clean)
            return abs(pred_num - exp_num) < 0.001
        except ValueError:
            pass

        return False

    def run_ablation_study(self,
                          dataset_path: Path,
                          ground_truth_path: Path = None,
                          max_samples: int = 20) -> Dict[str, AccuracyResult]:
        """Run complete ablation study."""
        results = {}

        for condition_name in self.test_conditions:
            print(f"\n{'='*50}")
            print(f"Testing condition: {condition_name}")
            print(f"{'='*50}")

            try:
                result = self.test_condition(
                    condition_name, dataset_path, ground_truth_path, max_samples
                )
                results[condition_name] = result

                print(f"Results for {condition_name}:")
                print(f"  Accuracy: {result.accuracy:.1%}")
                print(f"  Correct: {result.correct_answers}/{result.total_samples}")
                print(f"  Failed: {result.failed_traces}")

            except Exception as e:
                logger.error(f"Failed to test condition {condition_name}: {e}")

        return results

    def compare_conditions(self, results: Dict[str, AccuracyResult]) -> str:
        """Create comparison report."""
        if not results:
            return "No results to compare"

        report = "\nAblation Study Results Summary:\n"
        report += "=" * 50 + "\n"

        # Sort by accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1].accuracy, reverse=True)

        for condition, result in sorted_results:
            report += f"{condition:15} | "
            report += f"Accuracy: {result.accuracy:6.1%} | "
            report += f"Correct: {result.correct_answers:2}/{result.total_samples:2} | "
            report += f"Failed: {result.failed_traces:2}\n"

        # Best vs worst comparison
        if len(sorted_results) >= 2:
            best = sorted_results[0]
            worst = sorted_results[-1]
            improvement = best[1].accuracy - worst[1].accuracy
            report += f"\nBest condition ({best[0]}) vs Worst ({worst[0]}): "
            report += f"{improvement:+.1%} improvement\n"

        return report

    def save_results(self, results: Dict[str, AccuracyResult], output_path: Path):
        """Save detailed results to JSON."""
        serializable_results = {}
        for condition, result in results.items():
            serializable_results[condition] = {
                "condition_name": result.condition_name,
                "accuracy": result.accuracy,
                "total_samples": result.total_samples,
                "correct_answers": result.correct_answers,
                "failed_traces": result.failed_traces,
                "details": result.details
            }

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    tester = AccuracyTester()

    # Run ablation study
    results = tester.run_ablation_study(
        dataset_path=Path("data/clevr_easy_subset/subset.csv"),
        ground_truth_path=None,  # Add ground truth file if available
        max_samples=10  # Start small for testing
    )

    # Print comparison
    print(tester.compare_conditions(results))

    # Save detailed results
    tester.save_results(results, Path("evaluation/ablation_results.json"))
