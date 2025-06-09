#!/usr/bin/env python3
"""
Example usage of the spatial_trace package.

This script demonstrates how to use the refactored spatial reasoning pipeline
to process CLEVR dataset questions with spatial reasoning traces.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Import the spatial_trace package
from spatial_trace.utils import read_csv_data
from spatial_trace.tools import tool_registry
from spatial_trace import SpatialReasoningPipeline, TraceProcessor

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
    """Main example function."""
    print("Spatial Trace Package Example")
    print("=" * 40)

    setup_environment()    
    print("\n1. Initializing Spatial Reasoning Pipeline...")
    pipeline = SpatialReasoningPipeline(max_steps=10)
    
    print("\n2. Checking System Status...")
    status = pipeline.check_system_status()
    print(f"   LLM Available: {'✓' if status['llm_available'] else '✗'}")
    print(f"   Available Tools: {status['available_tools']}")
    
    if not status['llm_available']:
        print("   ERROR: LLM client not available. Please check your OpenAI API key.")
        return
    
    # Try to load CLEVR dataset
    script_dir = Path(__file__).resolve().parent
    csv_file_path = script_dir / "data" / "clevr_easy_subset" / "subset.csv"
    
    if not csv_file_path.exists():
        print(f"\n3. CLEVR dataset not found at: {csv_file_path}")
        print("   Creating a simple example instead...")
        
        # Simple example without dataset
        question = "Is the red block to the left of the blue sphere?"
        # You would need to provide an actual image path here
        print(f"   Example question: {question}")
        print("   (You would need to provide an actual image path to run this)")
        return
    
    print(f"\n3. Loading CLEVR dataset from: {csv_file_path}")
    clevr_data = read_csv_data(csv_file_path)
    
    if clevr_data is None:
        print("   ERROR: Failed to load dataset")
        return
    
    print(f"   Loaded {len(clevr_data)} samples")
    
    # Process the first entry
    print("\n4. Processing First Dataset Entry...")
    # Specifices dataset entry
    first_entry = clevr_data.iloc[1]
    
    question = first_entry['question']
    image_path = script_dir / "data" / "clevr_easy_subset" / first_entry['image_path']
    
    print(f"   Question: {question}")
    print(f"   Image: {image_path}")
    
    if not image_path.exists():
        print(f"   ERROR: Image not found at {image_path}")
        return
    
    # Generate reasoning trace
    print("\n5. Generating Reasoning Trace...")
    trace = pipeline.generate_reasoning_trace(question, image_path)
    
    if not trace:
        print("   ERROR: Failed to generate reasoning trace")
        return
    
    print(f"   Generated trace with {len(trace)} messages")
    
    # Process the trace
    print("\n6. Processing Trace Results...")
    processor = TraceProcessor()
    
    # Extract final answer
    final_answer = processor.extract_final_answer(trace)
    print(f"   Final Answer: {final_answer or 'No answer reached'}")
    
    # Analyze trace
    analysis = processor.analyze_trace(trace)
    print(f"   Tool calls made: {analysis['tool_call_count']}")
    print(f"   Reasoning steps: {analysis['reasoning_step_count']}")
    
    if analysis.get('tool_usage'):
        print("   Tool usage:")
        for tool, count in analysis['tool_usage'].items():
            print(f"     - {tool}: {count} times")
    
    # Save trace
    output_path = script_dir / "example_trace.json"
    success = processor.save_trace_with_analysis(
        trace, output_path, question, image_path
    )
    
    if success:
        print(f"\n7. Trace saved to: {output_path}")
    else:
        print("\n7. ERROR: Failed to save trace")
    
    # Print trace summary
    print("\n8. Trace Summary:")
    summary = processor.create_trace_summary(trace)
    print(summary)
    
    print("\n" + "=" * 40)
    print("Example completed successfully!")
    print("=" * 40)


if __name__ == "__main__":
    main() 