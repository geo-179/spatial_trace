#!/usr/bin/env python3
"""
Example usage of the spatial_trace package.

This script demonstrates how to use the refactored spatial reasoning pipeline
to process CLEVR dataset questions with spatial reasoning traces.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from tap import Tap
from spatial_trace.utils import read_csv_data
from spatial_trace.tools import tool_registry
from spatial_trace import SpatialReasoningPipeline, TraceProcessor

# First terminal: verify = false
# Second terminal: verify = true

class Arguments(Tap):
    """Command line arguments for the spatial trace demo."""
    max_steps: int = 10  # Maximum number of reasoning steps
    # question_idx: int = 1  # Index of question to process from dataset
    verify: bool = False
    number_questions: int = 100 # Number of questions to process from dataset
    data_dir: str = "data/clevr_human_subset"  # Path to data directory
    output_file: str = "example_trace.json"  # Output file for trace results

def setup_environment():
    """Set up environment variables for tools (if not already set)."""
    current_dir = Path(__file__).resolve().parent
    env_file = current_dir / ".env"

    if env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded environment variables from: {env_file}")
    else:
        print(f"No .env file found at: {env_file}")

    if os.getenv("OPENAI_API_KEY"):
        print("✓ OpenAI API key found")
    else:
        print("✗ OpenAI API key not found")

    sam2_path = current_dir.parent / "sam2"
    dav2_path = current_dir.parent / "Depth-Anything-V2"

    if not os.getenv("SAM2_PATH") and sam2_path.exists():
        os.environ["SAM2_PATH"] = str(sam2_path)
        print(f"Set SAM2_PATH to: {sam2_path}")

    if not os.getenv("DAV2_PATH") and dav2_path.exists():
        os.environ["DAV2_PATH"] = str(dav2_path)
        print(f"Set DAV2_PATH to: {dav2_path}")

    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY still not set.")
        print("Please either:")
        print("1. Create a .env file with: OPENAI_API_KEY=your-api-key")
        print("2. Export it: export OPENAI_API_KEY='your-api-key'")


def main():
    """Main example function with verification."""
    args = Arguments().parse_args()

    print("\nSpatial Trace Package Example with Verification")
    print("=" * 50)

    setup_environment()
    print("\nStep 1: Initializing Spatial Reasoning Pipeline with Verification")

    pipeline = SpatialReasoningPipeline(
        max_steps=args.max_steps,
        enable_verification=args.verify,
        min_acceptable_rating=6.0,
        max_regeneration_attempts=2
    )

    print("\nStep 2: Checking System Status")
    status = pipeline.check_system_status()
    print(f"• LLM Available: {'✓' if status['llm_available'] else '✗'}")
    print(f"• Available Tools: {status['available_tools']}")

    if not status['llm_available']:
        print("\nERROR: LLM client not available. Please check your OpenAI API key.")
        return

    script_dir = Path(__file__).resolve().parent
    csv_file_path = script_dir / args.data_dir / "subset.csv"

    if not csv_file_path.exists():
        print(f"\nERROR: Dataset not found at: {csv_file_path}")
        return

    print(f"\nStep 3: Loading CLEVR dataset from: {csv_file_path}")
    clevr_data = read_csv_data(csv_file_path)

    if clevr_data is None:
        print("\nERROR: Failed to load dataset")
        return

    print(f"• Loaded {len(clevr_data)} samples")

    print("\nStep 4: Processing Dataset Entry")

    total = 0
    correct = 0

    for question_idx in range(args.number_questions):
        first_entry = clevr_data.iloc[question_idx]

        question = first_entry['question']
        answer = first_entry['answer']
        image_path = script_dir / args.data_dir / first_entry['image_path']

        print(f"• Question: {question}")
        print(f"• Image: {image_path}")

        if not image_path.exists():
            print(f"\nERROR: Image not found at {image_path}")
            return

        print("\nStep 5: Generating Reasoning Trace with Verification")
        trace = pipeline.generate_reasoning_trace(question, image_path)

        if not trace:
            print("\nERROR: Failed to generate reasoning trace")
            return

        print(f"• Generated trace with {len(trace)} messages")

        verification_history = pipeline.get_verification_history()
        if verification_history:
            print(f"• Verification performed on {len(verification_history)} steps")
            avg_rating = sum(v["result"]["rating"] for v in verification_history) / len(verification_history)
            print(f"• Average step rating: {avg_rating:.1f}/10")

        print("\nStep 6: Processing Trace Results")
        processor = TraceProcessor()

        final_answer = processor.extract_final_answer(trace)
        print(f"• Final Answer: {final_answer or 'No answer reached'}")

        if final_answer is None:
            continue

        final_answer = final_answer.lower()

        if final_answer == answer:
            correct += 1
        total += 1

        analysis = processor.analyze_trace(trace)
        print(f"• Tool calls made: {analysis['tool_call_count']}")
        print(f"• Reasoning steps: {analysis['reasoning_step_count']}")

        if analysis.get('tool_usage'):
            print("\nTool Usage:")
            for tool, count in analysis['tool_usage'].items():
                print(f"• {tool}: {count} times")

        output_path = script_dir / args.output_file
        success = processor.save_trace_with_analysis(
            trace, output_path, question, image_path, verification_history
        )

        if success:
            print(f"\nStep 7: Saving Results")
            print(f"Trace data saved to: {output_path}")

            verification_path = script_dir / f"verification_history_{question_idx}.json"
            if pipeline.save_verification_history(verification_path):
                print(f"• Verification history saved to: {verification_path}")
        else:
            print("\nERROR: Failed to save trace")

        print("\nStep 8: Trace Summary")
        summary = processor.create_trace_summary(trace)
        print(summary)

        print("\n" + "=" * 50)
        print("Example completed successfully!")
        print("=" * 50)

    print(f"Correctness: {correct} out of {total}")
    print(f"{100 * correct/total}% correct")


if __name__ == "__main__":
    main()
