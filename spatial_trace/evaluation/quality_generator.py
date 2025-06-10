"""
Generate high-quality reasoning traces for evaluation.
"""
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging

# Adjust import path based on your package structure
from spatial_trace import SpatialReasoningPipeline
from spatial_trace.utils import read_csv_data

logger = logging.getLogger(__name__)


class QualityGenerator:
    """Generates quality traces with high verification standards."""

    def __init__(self, min_rating: float = 8.0):
        """Initialize with strict quality requirements."""
        self.min_rating = min_rating
        self.pipeline = SpatialReasoningPipeline(
            enable_verification=True,
            min_acceptable_rating=min_rating,
            max_regeneration_attempts=3
        )

    def generate_quality_dataset(self,
                                input_csv: Path,
                                output_path: Path,
                                max_samples: int = 50) -> Dict[str, Any]:
        """
        Generate high-quality traces from CLEVR dataset.

        Args:
            input_csv: Path to CLEVR CSV dataset
            output_path: Where to save quality traces
            max_samples: Maximum number of samples to process

        Returns:
            Statistics about generation process
        """
        logger.info(f"Loading dataset from {input_csv}")
        data = read_csv_data(input_csv)
        if data is None:
            raise ValueError(f"Could not load dataset from {input_csv}")

        quality_traces = []
        stats = {
            "total_attempted": 0,
            "high_quality_generated": 0,
            "failed_generations": 0,
            "average_rating": 0.0,
            "ratings": []
        }

        # Process samples
        for idx, row in data.head(max_samples).iterrows():
            stats["total_attempted"] += 1

            try:
                question = row['question']
                image_path = Path(input_csv).parent / row['image_path']

                logger.info(f"Processing sample {idx}: {question}")

                # Generate trace
                trace = self.pipeline.generate_reasoning_trace(question, image_path)

                if trace:
                    # Get verification history
                    verification_history = self.pipeline.get_verification_history()

                    if verification_history:
                        # Calculate average rating for this trace
                        ratings = [v["result"]["rating"] for v in verification_history]
                        avg_rating = sum(ratings) / len(ratings)
                        stats["ratings"].append(avg_rating)

                        # Only keep high-quality traces
                        if avg_rating >= self.min_rating:
                            quality_traces.append({
                                "id": f"trace_{idx}",
                                "question": question,
                                "image_path": str(image_path),
                                "trace": trace,
                                "verification_history": verification_history,
                                "average_rating": avg_rating
                            })
                            stats["high_quality_generated"] += 1
                            logger.info(f"High-quality trace generated (rating: {avg_rating:.1f})")
                        else:
                            logger.warning(f"Trace quality too low (rating: {avg_rating:.1f})")
                    else:
                        logger.warning("No verification history available")
                else:
                    stats["failed_generations"] += 1
                    logger.error(f"Failed to generate trace for sample {idx}")

            except Exception as e:
                stats["failed_generations"] += 1
                logger.error(f"Error processing sample {idx}: {e}")

        # Calculate final stats
        if stats["ratings"]:
            stats["average_rating"] = sum(stats["ratings"]) / len(stats["ratings"])

        # Save quality traces
        if quality_traces:
            with open(output_path, 'w') as f:
                json.dump(quality_traces, f, indent=2, default=str)
            logger.info(f"Saved {len(quality_traces)} quality traces to {output_path}")

        return stats


if __name__ == "__main__":
    generator = QualityGenerator(min_rating=8.0)

    stats = generator.generate_quality_dataset(
        Path("data/clevr_easy_subset/subset.csv"),
        Path("evaluation/quality_traces.json"),
        max_samples=10  # Start small for testing
    )

    print("Quality Generation Results:")
    print(f"Attempted: {stats['total_attempted']}")
    print(f"High Quality Generated: {stats['high_quality_generated']}")
    print(f"Failed: {stats['failed_generations']}")
    print(f"Average Rating: {stats['average_rating']:.2f}")
