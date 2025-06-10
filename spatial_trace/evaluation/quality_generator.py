"""
Generate high-quality reasoning traces for evaluation with organized output structure.
"""
import json
import pandas as pd
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import argparse

# Fix import path - add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Now import from the correct location
from spatial_trace.inference import SpatialReasoningPipeline
from spatial_trace.evaluation.utils import read_csv_data

logger = logging.getLogger(__name__)

class QualityGenerator:
    """Generates quality traces with organized experiment structure."""

    def __init__(self,
                 min_rating: float = 8.0,
                 experiment_name: str = None,
                 default_max_samples: int = 50,
                 enable_regeneration: bool = True):
        """Initialize with quality requirements and experiment name.

        Args:
            min_rating: Minimum rating threshold for acceptable traces.
            experiment_name: Name of the experiment.
            default_max_samples: Default number of samples to process.
            enable_regeneration: If False, get ratings but do not perform correction.
        """
        self.min_rating = min_rating
        self.default_max_samples = default_max_samples
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.enable_regeneration = enable_regeneration

        # Verification is always ON to get ratings. Regeneration is the key switch.
        self.pipeline = SpatialReasoningPipeline(
            enable_verification=True,
            enable_regeneration=self.enable_regeneration,
            min_acceptable_rating=min_rating,
            max_regeneration_attempts=5
        )

        # Base experiments directory - CREATE LOCALLY in evaluation/experiments/
        self.experiments_base = Path(__file__).parent / "experiments"
        self.experiment_dir = self.experiments_base / self.experiment_name

        print(f"Initialized QualityGenerator for experiment: {self.experiment_name}")
        print(f"Default max samples: {self.default_max_samples}")
        print(f"Experiments will be saved to: {self.experiments_base.absolute()}")

        self.successful_verifications = []  # Cache of recent successful verifications

    def generate_quality_dataset(self,
                                input_csv: Path,
                                max_samples: int = None) -> Dict[str, Any]:
        """
        Generate high-quality traces from mixed CLEVR dataset with organized structure.

        Args:
            input_csv: Path to mixed CLEVR CSV dataset
            max_samples: Maximum number of samples to process (uses default if None)

        Returns:
            Statistics about generation process
        """
        # Use default if not specified
        if max_samples is None:
            max_samples = self.default_max_samples

        print(f"Loading CLEVR dataset: {max_samples} samples")

        data = read_csv_data(input_csv)
        if data is None:
            raise ValueError(f"Could not load dataset from {input_csv}")

        # Log dataset info
        total_available = len(data)
        print(f"Dataset contains {total_available} samples")

        # Create experiment directory structure
        self._setup_experiment_structure()

        quality_traces = []
        stats = {
            "experiment_name": self.experiment_name,
            "dataset_info": {
                "total_available_samples": total_available,
                "requested_samples": max_samples,
                "processed_samples": min(max_samples, total_available)
            },
            "total_attempted": 0,
            "high_quality_generated": 0,
            "failed_generations": 0,
            "average_rating": 0.0,
            "ratings": [],
            "processed_questions": []
        }

        # Process samples (take top N)
        for idx, row in data.head(max_samples).iterrows():
            stats["total_attempted"] += 1

            try:
                question = row['question']
                answer = row['answer']
                difficulty = row.get('difficulty', 'unknown')
                image_filename = row['image_filename']

                # CORRECTED LOGIC: Construct path from filename, not from the buggy 'image_path' column.
                image_path = Path(input_csv).parent / 'images' / image_filename

                print(f"Processing {idx+1}/{max_samples}: {question[:50]}...")

                # Create question-specific directory
                question_dir = self._create_question_directory(idx, question, image_filename)

                # Copy original image to question directory
                original_image_dest = question_dir / "images" / image_filename
                shutil.copy2(image_path, original_image_dest)

                # Generate trace
                trace = self.pipeline.generate_reasoning_trace(question, image_path)

                if trace:
                    # Always get history and tool images
                    verification_history = self.pipeline.get_verification_history()
                    tool_images = self._extract_and_save_tool_images(trace, question_dir)

                    avg_rating = 0.0
                    if verification_history:
                        ratings = [v["result"]["rating"] for v in verification_history
                                 if "result" in v and "rating" in v["result"]]
                        if ratings:
                            avg_rating = sum(ratings) / len(ratings)
                            stats["ratings"].append(avg_rating)

                    # Create the data dictionary regardless of verification
                    trace_data = {
                        "id": f"trace_{idx}",
                        "question_index": idx,
                        "question": question,
                        "expected_answer": answer,
                        "difficulty": difficulty,
                        "image_filename": image_filename,
                        "original_image_path": str(original_image_dest.relative_to(question_dir)),
                        "trace": trace,
                        "verification_history": verification_history,
                        "average_rating": avg_rating,
                        "tool_images": tool_images,
                        "generation_timestamp": datetime.now().isoformat()
                    }

                    # Always save the files for this question.
                    self._save_question_files(question_dir, trace_data)

                    # Now, decide what to print and whether to include it in the final summary.
                    if not self.enable_regeneration or (avg_rating >= self.min_rating):
                        # If regeneration is OFF, we accept everything.
                        # If regeneration is ON, we only accept traces that meet the rating.
                        quality_traces.append(trace_data)
                        stats["high_quality_generated"] += 1
                        if not self.enable_regeneration:
                            print(f"✓ Trace generated for analysis (rating: {avg_rating:.1f}, regeneration disabled)")
                        else:
                            print(f"✓ High-quality trace (rating: {avg_rating:.1f})")
                    else:
                        # This code is only reachable if regeneration was ON and the rating was too low.
                        print(f"✗ Low quality (rating: {avg_rating:.1f})")

                    # Track processed question
                    stats["processed_questions"].append({
                        "index": idx,
                        "question": question,
                        "answer": answer,
                        "difficulty": difficulty,
                        "rating": avg_rating if self.enable_regeneration else None,
                        "directory": str(question_dir.relative_to(self.experiment_dir))
                    })
                else:
                    stats["failed_generations"] += 1
                    print(f"✗ Failed to generate trace")

            except Exception as e:
                stats["failed_generations"] += 1
                print(f"✗ Error: {e}")

        # Calculate final stats
        if stats["ratings"]:
            stats["average_rating"] = sum(stats["ratings"]) / len(stats["ratings"])

        # Final summary
        print(f"\nResults: {stats['high_quality_generated']}/{stats['total_attempted']} high-quality traces")
        print(f"Success rate: {stats['high_quality_generated']/stats['total_attempted']*100:.1f}%")
        if stats["ratings"]:
            print(f"Average rating: {stats['average_rating']:.1f}")

        # Save experiment summary
        self._save_experiment_summary(stats, quality_traces)

        return stats

    def _setup_experiment_structure(self):
        """Create the base experiment directory structure."""
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.experiment_dir / "questions").mkdir(exist_ok=True)
        (self.experiment_dir / "summary").mkdir(exist_ok=True)

        print(f"Created experiment structure at {self.experiment_dir}")

    def _create_question_directory(self, idx: int, question: str, image_filename: str) -> Path:
        """Create directory structure for a specific question."""
        # Sanitize question for directory name (keep it short and safe)
        safe_question = question[:50].replace('?', '').replace('/', '_').replace('\\', '_')
        safe_question = ''.join(c for c in safe_question if c.isalnum() or c in (' ', '-', '_')).strip()

        question_dir_name = f"q{idx:03d}_{safe_question}_{image_filename.replace('.png', '')}"
        question_dir = self.experiment_dir / "questions" / question_dir_name

        # Create subdirectories
        question_dir.mkdir(parents=True, exist_ok=True)
        (question_dir / "images").mkdir(exist_ok=True)
        (question_dir / "traces").mkdir(exist_ok=True)
        (question_dir / "verification").mkdir(exist_ok=True)

        logger.debug(f"Created question directory: {question_dir}")
        return question_dir

    def _extract_and_save_tool_images(self, trace: List[Dict[str, Any]], question_dir: Path) -> List[Dict[str, str]]:
        """Extract and save tool images using pipeline tracking."""

        # Get tool images from pipeline
        pipeline_tool_images = self.pipeline.get_current_trace_tool_images()

        if not pipeline_tool_images:
            logger.info("No tool images found in pipeline tracking")
            return []

        saved_tool_images = []

        for tool_info in pipeline_tool_images:
            source_path = Path(tool_info["source_path"])

            if not source_path.exists():
                logger.warning(f"Tool image not found: {source_path}")
                continue

            # Create simple, descriptive filename
            step = tool_info["step_index"]
            attempt = tool_info["attempt"]
            tool_name = tool_info["tool_name"].lower()

            dest_filename = f"step_{step}_attempt_{attempt}_{tool_name}_result.png"
            dest_path = question_dir / "images" / dest_filename

            try:
                # Copy tool image to question directory
                shutil.copy2(source_path, dest_path)

                saved_info = {
                    "step_index": step,
                    "attempt": attempt,
                    "tool_name": tool_info["tool_name"],
                    "filename": dest_filename,
                    "path": f"images/{dest_filename}",  # Relative to question dir
                    "reasoning": tool_info["reasoning"],
                    "source_path": str(source_path),
                    "timestamp": tool_info["timestamp"]
                }

                saved_tool_images.append(saved_info)
                logger.info(f"Saved tool image: {dest_path}")

            except Exception as e:
                logger.error(f"Failed to copy tool image {source_path} to {dest_path}: {e}")

        logger.info(f"Successfully saved {len(saved_tool_images)} tool images")
        return saved_tool_images

    def _save_question_files(self, question_dir: Path, trace_data: Dict[str, Any]):
        """Enhanced to save comprehensive verification data and update trace image references."""

        # Update trace to reference local image copies
        updated_trace = self._update_trace_image_references(trace_data["trace"], trace_data["tool_images"])
        trace_data["trace"] = updated_trace

        # Save complete trace data
        trace_file = question_dir / "traces" / "complete_trace.json"
        with open(trace_file, 'w') as f:
            json.dump(trace_data, f, indent=2, default=str)

        # Save just the trace messages for easier parsing
        messages_file = question_dir / "traces" / "messages.json"
        with open(messages_file, 'w') as f:
            json.dump(trace_data["trace"], f, indent=2, default=str)

        # Enhanced verification history with summary
        if trace_data.get("verification_history"):
            verification_dir = question_dir / "verification"

            # Save full verification history
            verification_file = verification_dir / "verification_history.json"
            with open(verification_file, 'w') as f:
                json.dump(trace_data["verification_history"], f, indent=2, default=str)

            # Save verification summary
            verification_summary = self._create_verification_summary(trace_data["verification_history"])
            summary_file = verification_dir / "verification_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(verification_summary, f, indent=2, default=str)

        # Save question metadata
        metadata = {
            "question_index": trace_data["question_index"],
            "question": trace_data["question"],
            "expected_answer": trace_data["expected_answer"],
            "difficulty": trace_data["difficulty"],
            "image_filename": trace_data["image_filename"],
            "average_rating": trace_data["average_rating"],
            "tool_images": trace_data["tool_images"],
            "generation_timestamp": trace_data["generation_timestamp"]
        }
        metadata_file = question_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.debug(f"Saved question files to {question_dir}")

    def _create_verification_summary(self, verification_history: List[Dict]) -> Dict[str, Any]:
        """Create a summary of verification attempts."""
        if not verification_history:
            return {}

        attempts_by_step = {}
        for entry in verification_history:
            step_idx = entry["step_index"]
            if step_idx not in attempts_by_step:
                attempts_by_step[step_idx] = []
            attempts_by_step[step_idx].append(entry)

        summary = {
            "total_verification_attempts": len(verification_history),
            "unique_steps_verified": len(attempts_by_step),
            "steps_requiring_regeneration": sum(1 for step_attempts in attempts_by_step.values()
                                              if len(step_attempts) > 1),
            "average_rating": sum(entry["result"]["rating"] for entry in verification_history) / len(verification_history),
            "rating_distribution": {},
            "step_details": {}
        }

        # Rating distribution
        for entry in verification_history:
            rating = entry["result"]["rating"]
            rating_bucket = f"{int(rating)}-{int(rating)+1}"
            summary["rating_distribution"][rating_bucket] = summary["rating_distribution"].get(rating_bucket, 0) + 1

        # Step details
        for step_idx, attempts in attempts_by_step.items():
            summary["step_details"][f"step_{step_idx}"] = {
                "attempts": len(attempts),
                "final_rating": attempts[-1]["result"]["rating"],
                "improvement": attempts[-1]["result"]["rating"] - attempts[0]["result"]["rating"] if len(attempts) > 1 else 0,
                "regeneration_needed": len(attempts) > 1
            }

        return summary

    def _save_experiment_summary(self, stats: Dict[str, Any], quality_traces: List[Dict[str, Any]]):
        """Save experiment summary and high-quality traces."""
        summary_dir = self.experiment_dir / "summary"

        # Save statistics
        stats_file = summary_dir / "experiment_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)

        # Save high-quality traces summary
        if quality_traces:
            quality_file = summary_dir / "high_quality_traces.json"
            with open(quality_file, 'w') as f:
                json.dump(quality_traces, f, indent=2, default=str)

        # Save experiment configuration
        config = {
            "experiment_name": self.experiment_name,
            "min_rating_threshold": self.min_rating,
            "pipeline_config": self.pipeline.check_system_status(),
            "generation_timestamp": datetime.now().isoformat()
        }
        config_file = summary_dir / "experiment_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)

        print(f"Saved experiment summary to {summary_dir}")

    def set_experiment_name(self, name: str):
        """Change the experiment name (before running generation)."""
        self.experiment_name = name
        self.experiment_dir = self.experiments_base / name
        print(f"Changed experiment name to: {name}")

    def _update_trace_image_references(self, trace: List[Dict[str, Any]], tool_images: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Update trace messages to reference local image copies instead of original tool paths."""

        # Create mapping from both original tool output paths and safe copy paths to local paths
        path_mapping = {}
        for tool_image in tool_images:
            safe_copy_path = tool_image["source_path"]
            local_path = tool_image["path"]  # Already relative to question dir

            # Map the safe copy path
            path_mapping[safe_copy_path] = local_path

            # Also map the original tool output path (extract from safe copy path)
            # Safe copy format: /path/to/tool/toolname_step_attempt_timestamp.png
            # Original format: /path/to/tool/original_filename.png
            safe_copy_path_obj = Path(safe_copy_path)
            tool_dir = safe_copy_path_obj.parent
            tool_name = tool_image["tool_name"].lower()

            # Determine original filename based on tool
            if tool_name == "sam2":
                original_filename = "segmented_image.png"
            elif tool_name == "dav2":
                original_filename = "dav2_result.png"
            elif tool_name == "trellis":
                original_filename = "novel_view3_topdown.png"
            else:
                # Generic fallback - try to extract from reasoning or use tool name
                original_filename = f"{tool_name}_output.png"

            original_tool_path = tool_dir / original_filename
            path_mapping[str(original_tool_path)] = local_path

        # Update trace messages
        updated_trace = []
        for message in trace:
            if message.get("role") == "user":
                content = message.get("content")

                # Handle both message formats
                if isinstance(content, str) and "Tool output:" in content:
                    # Simple format: content is a string
                    updated_content = content
                    for original_path, local_path in path_mapping.items():
                        if original_path in updated_content:
                            updated_content = updated_content.replace(original_path, local_path)
                            break

                    updated_message = message.copy()
                    updated_message["content"] = updated_content
                    updated_trace.append(updated_message)

                elif isinstance(content, list):
                    # Complex format: content is a list with text and image components
                    updated_content = []
                    for item in content:
                        if item.get("type") == "text" and "Tool output:" in item.get("text", ""):
                            # Update the text component
                            updated_text = item["text"]
                            for original_path, local_path in path_mapping.items():
                                if original_path in updated_text:
                                    updated_text = updated_text.replace(original_path, local_path)
                                    break

                            updated_item = item.copy()
                            updated_item["text"] = updated_text
                            updated_content.append(updated_item)
                        else:
                            # Keep other components unchanged (like images)
                            updated_content.append(item)

                    updated_message = message.copy()
                    updated_message["content"] = updated_content
                    updated_trace.append(updated_message)
                else:
                    # Not a tool output message, keep unchanged
                    updated_trace.append(message)
            else:
                # Not a user message, keep unchanged
                updated_trace.append(message)

        return updated_trace


def read_csv_data(csv_path: Path) -> Optional[pd.DataFrame]:
    """Load data from CSV file."""
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Error loading CSV from {csv_path}: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate high-quality reasoning traces")
    parser.add_argument("--dataset", type=str, default="clevr_human_subset",
                       help="Name of the dataset folder (default: clevr_human_subset)")
    parser.add_argument("--experiment", type=str, required=True,
                       help="Name of the experiment (output directory name)")
    parser.add_argument("--max_samples", type=int, default=10,
                       help="Maximum number of samples to process (default: 10)")
    parser.add_argument("--min_rating", type=float, default=8.0,
                       help="Minimum quality rating to accept (default: 8.0)")

    # This flag now correctly controls REGENERATION.
    parser.add_argument('--no-verification', dest='regeneration', action='store_false',
                       help="Disables the regeneration loop. Ratings will still be generated for analysis.")
    parser.set_defaults(regeneration=True)

    args = parser.parse_args()

    # Construct an absolute path to the data directory relative to this script's location.
    # This makes pathing independent of the current working directory.
    try:
        # Assumes the 'data' directory is three levels up from this script file.
        # .../evaluation/quality_generator.py -> .../evaluation -> .../spatial_trace -> project_root
        project_root = Path(__file__).resolve().parent.parent.parent
        input_csv = project_root / 'data' / args.dataset / 'subset.csv'

        if not input_csv.exists():
            print(f"FATAL: Dataset CSV file not found.")
            print(f"Attempted to find it at: {input_csv}")
            print(f"Please ensure the 'data' directory is in the project root: {project_root}")
            exit(1)

    except Exception as e:
        print(f"FATAL: Could not construct a valid path to the dataset. Error: {e}")
        exit(1)


    # Create generator, passing the regeneration flag.
    generator = QualityGenerator(
        min_rating=args.min_rating,
        experiment_name=args.experiment,
        default_max_samples=args.max_samples,
        enable_regeneration=args.regeneration
    )

    # Run the generation
    stats = generator.generate_quality_dataset(
        input_csv,
        max_samples=args.max_samples
    )

    print("\nQuality Generation Results:")
    print(f"Dataset: {args.dataset}")
    print(f"Experiment: {args.experiment}")
    print(f"Verification: {'Enabled' if args.regeneration else 'Disabled'}")
    print(f"Output Directory: {generator.experiment_dir}")
    print(f"Attempted: {stats['total_attempted']}")
    print(f"High Quality Generated: {stats['high_quality_generated']}")
    print(f"Failed: {stats['failed_generations']}")
    print(f"Average Rating: {stats['average_rating']:.2f}")
