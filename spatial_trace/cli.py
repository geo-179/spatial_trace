"""
Command-line interface for spatial reasoning pipeline.
"""
import argparse
import sys
import json
from pathlib import Path
from typing import Optional

from .inference import SpatialReasoningPipeline, TraceProcessor
from .utils import read_csv_data, get_logger, set_log_level

logger = get_logger()


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Spatial Trace: LLM-based spatial reasoning pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single question inference
  spatial-trace reason --question "Is the red block to the left of the blue sphere?" --image image.jpg
  
  # Process a dataset
  spatial-trace batch --input dataset.csv --output-dir results/
  
  # Check system status
  spatial-trace status
        """
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    reason_parser = subparsers.add_parser("reason", help="Run spatial reasoning on a single question")
    reason_parser.add_argument("--question", required=True, help="Spatial reasoning question")
    reason_parser.add_argument("--image", required=True, type=Path, help="Path to input image")
    reason_parser.add_argument("--output", type=Path, help="Path to save reasoning trace (optional)")
    reason_parser.add_argument("--max-steps", type=int, default=10, help="Maximum reasoning steps")
    
    batch_parser = subparsers.add_parser("batch", help="Process a dataset of questions")
    batch_parser.add_argument("--input", required=True, type=Path, help="Path to CSV dataset")
    batch_parser.add_argument("--output-dir", required=True, type=Path, help="Directory to save results")
    batch_parser.add_argument("--max-steps", type=int, default=10, help="Maximum reasoning steps")
    batch_parser.add_argument("--question-column", default="question", help="Name of question column in CSV")
    batch_parser.add_argument("--image-column", default="image_path", help="Name of image path column in CSV")
    batch_parser.add_argument("--limit", type=int, help="Limit number of samples to process")
    
    status_parser = subparsers.add_parser("status", help="Check system status")
    
    return parser


def run_single_reasoning(args) -> int:
    """Run spatial reasoning on a single question."""
    try:
        logger.info(f"Running spatial reasoning on: {args.question}")
        logger.info(f"Image: {args.image}")
        
        pipeline = SpatialReasoningPipeline(max_steps=args.max_steps)
        
        status = pipeline.check_system_status()
        if not status["llm_available"]:
            logger.error("LLM client is not available. Please check your configuration.")
            return 1
        
        trace = pipeline.generate_reasoning_trace(args.question, args.image)
        
        if not trace:
            logger.error("Failed to generate reasoning trace")
            return 1
        
        processor = TraceProcessor()
        final_answer = processor.extract_final_answer(trace)
        
        print("\n" + "="*50)
        print("SPATIAL REASONING RESULT")
        print("="*50)
        print(f"Question: {args.question}")
        print(f"Image: {args.image}")
        print(f"Final Answer: {final_answer or 'No answer reached'}")
        print("="*50)
        
        if args.output:
            success = processor.save_trace_with_analysis(
                trace, args.output, args.question, args.image
            )
            if success:
                logger.info(f"Reasoning trace saved to: {args.output}")
            else:
                logger.error(f"Failed to save trace to: {args.output}")
                return 1
        
        summary = processor.create_trace_summary(trace)
        print(f"\n{summary}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during reasoning: {e}")
        return 1


def run_batch_processing(args) -> int:
    """Process a dataset of questions."""
    try:
        logger.info(f"Processing dataset: {args.input}")
        logger.info(f"Output directory: {args.output_dir}")
        
        df = read_csv_data(args.input)
        if df is None:
            logger.error(f"Failed to load dataset from {args.input}")
            return 1
        
        if args.question_column not in df.columns:
            logger.error(f"Question column '{args.question_column}' not found in dataset")
            return 1
        
        if args.image_column not in df.columns:
            logger.error(f"Image column '{args.image_column}' not found in dataset")
            return 1
        
        if args.limit:
            df = df.head(args.limit)
            logger.info(f"Limited dataset to {len(df)} samples")
        
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        pipeline = SpatialReasoningPipeline(max_steps=args.max_steps)
        processor = TraceProcessor()
        
        status = pipeline.check_system_status()
        if not status["llm_available"]:
            logger.error("LLM client is not available. Please check your configuration.")
            return 1
        
        results = []
        successful = 0
        
        for idx, row in df.iterrows():
            try:
                question = row[args.question_column]
                image_path = Path(row[args.image_column])
                
                logger.info(f"Processing sample {idx + 1}/{len(df)}: {question[:50]}...")
                
                trace = pipeline.generate_reasoning_trace(question, image_path)
                
                if trace:
                    final_answer = processor.extract_final_answer(trace)
                    
                    output_file = args.output_dir / f"trace_{idx:04d}.json"
                    processor.save_trace_with_analysis(trace, output_file, question, image_path)
                    
                    results.append({
                        "index": idx,
                        "question": question,
                        "image_path": str(image_path),
                        "final_answer": final_answer,
                        "trace_file": str(output_file),
                        "success": True
                    })
                    successful += 1
                else:
                    results.append({
                        "index": idx,
                        "question": question,
                        "image_path": str(image_path),
                        "final_answer": None,
                        "trace_file": None,
                        "success": False
                    })
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                results.append({
                    "index": idx,
                    "question": question if 'question' in locals() else "Unknown",
                    "image_path": str(image_path) if 'image_path' in locals() else "Unknown",
                    "final_answer": None,
                    "trace_file": None,
                    "success": False,
                    "error": str(e)
                })
        
        summary_file = args.output_dir / "batch_results.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "input_file": str(args.input),
                "total_samples": len(df),
                "successful": successful,
                "failed": len(df) - successful,
                "results": results
            }, f, indent=2)
        
        logger.info(f"Batch processing completed: {successful}/{len(df)} successful")
        logger.info(f"Results summary saved to: {summary_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during batch processing: {e}")
        return 1


def check_system_status(args) -> int:
    """Check and display system status."""
    try:
        pipeline = SpatialReasoningPipeline()
        status = pipeline.check_system_status()
        
        print("\n" + "="*50)
        print("SPATIAL TRACE SYSTEM STATUS")
        print("="*50)
        
        print(f"LLM Available: {'✓' if status['llm_available'] else '✗'}")
        print(f"LLM Model: {status['llm_info']['model_name']}")
        print(f"LLM Provider: {status['llm_info'].get('provider', 'Unknown')}")
        
        print(f"\nAvailable Tools: {len(status['available_tools'])}")
        for tool in status['available_tools']:
            tool_available = status['tool_availability'].get(tool, False)
            print(f"  - {tool}: {'✓' if tool_available else '✗'}")
        
        print(f"\nMax Steps: {status['max_steps']}")
        print("="*50)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error checking system status: {e}")
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    set_log_level(args.log_level)
    
    if args.command == "reason":
        return run_single_reasoning(args)
    elif args.command == "batch":
        return run_batch_processing(args)
    elif args.command == "status":
        return check_system_status(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 