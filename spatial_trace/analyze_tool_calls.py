#!/usr/bin/env python3
"""
Tool Call Distribution Analysis Script

This script analyzes the distribution of tool calls used by an LLM when reasoning through CLEVR problems.
It processes all question subfolders, extracts tool calls from complete_trace.json files,
and generates a comprehensive analysis with visualizations.

Author: Assistant
Date: 2024
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import glob
from tqdm import tqdm
import sys
from pathlib import Path

def validate_directories(questions_dir, output_dir):
    """
    Validate that required directories exist and are accessible.
    
    Args:
        questions_dir (str): Path to questions directory
        output_dir (str): Path to output directory
        
    Returns:
        bool: True if directories are valid, False otherwise
    """
    if not os.path.exists(questions_dir):
        print(f"ERROR: Questions directory does not exist: {questions_dir}")
        return False
    
    if not os.path.isdir(questions_dir):
        print(f"ERROR: Questions path is not a directory: {questions_dir}")
        return False
    
    if not os.path.exists(output_dir):
        print(f"WARNING: Output directory does not exist, creating: {output_dir}")
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"ERROR: Could not create output directory: {e}")
            return False
    
    return True

def extract_tool_calls_from_trace(trace_data):
    """
    Extract tool calls from a trace JSON structure.
    
    Args:
        trace_data (dict): Parsed JSON data from complete_trace.json
        
    Returns:
        list: List of tool names found in the trace
    """
    tool_calls = []
    
    if not isinstance(trace_data, dict) or 'trace' not in trace_data:
        return tool_calls
    
    trace_array = trace_data['trace']
    if not isinstance(trace_array, list):
        return tool_calls
    
    for message in trace_array:
        if isinstance(message, dict) and message.get('role') == 'assistant':
            # The content field contains a JSON string that needs to be parsed
            content = message.get('content')
            if content and isinstance(content, str):
                try:
                    # Parse the JSON content
                    content_data = json.loads(content)
                    
                    # Check if this is a tool call
                    if (isinstance(content_data, dict) and 
                        content_data.get('action') == 'tool_call'):
                        
                        tool_name = content_data.get('tool_name')
                        if tool_name and isinstance(tool_name, str):
                            tool_calls.append(tool_name.strip())
                            
                except json.JSONDecodeError:
                    # Skip messages with malformed JSON content
                    continue
                except Exception:
                    # Skip any other parsing errors
                    continue
    
    return tool_calls

def process_question_folder(folder_path):
    """
    Process a single question folder to extract tool calls.
    
    Args:
        folder_path (str): Path to question folder
        
    Returns:
        tuple: (list of tool calls, error message if any)
    """
    trace_file = os.path.join(folder_path, "traces", "complete_trace.json")
    
    if not os.path.exists(trace_file):
        return [], f"Trace file not found: {trace_file}"
    
    try:
        # Handle large files by reading with proper encoding
        with open(trace_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tool_calls = extract_tool_calls_from_trace(data)
        return tool_calls, None
        
    except json.JSONDecodeError as e:
        return [], f"JSON decode error in {trace_file}: {e}"
    except Exception as e:
        return [], f"Error processing {trace_file}: {e}"

def create_visualization(tool_counter, output_dir):
    """
    Create and save visualization of tool call distribution.
    
    Args:
        tool_counter (Counter): Counter object with tool call counts
        output_dir (str): Directory to save outputs
        
    Returns:
        str: Path to saved histogram
    """
    if not tool_counter:
        print("No data to visualize")
        return None
    
    # Set seaborn style for academic publications
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.4)
    
    # Define consistent color mapping for each tool
    tool_color_map = {
        'TRELLIS': '#1f77b4',  # Blue
        'SAM 2': '#ff7f0e',    # Orange  
        'DAv2': '#2ca02c'      # Green
    }
    
    # Prepare data for plotting
    tools = list(tool_counter.keys())
    # Format tool names properly
    formatted_tools = []
    for tool in tools:
        if tool.lower() == 'trellis':
            formatted_tools.append('TRELLIS')
        elif tool.lower() == 'sam2':
            formatted_tools.append('SAM 2')
        elif tool.lower() == 'dav2':
            formatted_tools.append('DAv2')
        else:
            formatted_tools.append(tool)
    tools = formatted_tools
    counts = list(tool_counter.values())
    total_calls = sum(counts)
    
    # Create figure with appropriate size for academic papers
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Assign consistent colors based on tool names
    colors = [tool_color_map.get(tool, '#808080') for tool in tools]
    
    # Create bar chart with seaborn styling
    bars = ax.bar(tools, counts, 
                  color=colors,
                  edgecolor='white',
                  linewidth=0.8,
                  alpha=0.85)
    
    # Customize the plot with academic styling (no title)
    ax.set_xlabel('Tool Name', fontsize=18)
    ax.set_ylabel('Number of Calls', fontsize=18)
    
    # Improve x-axis labels - make tool names larger
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=11)
    
    # Add value labels on top of bars with percentages
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percentage = (height / total_calls) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01,
                f'{int(height)}\n({percentage:.1f}%)', 
                ha='center', va='bottom', 
                fontsize=14, fontweight='bold',
                color='black')
    
    # Customize grid for academic style - horizontal dotted lines only
    ax.grid(True)  # Turn on grid first
    ax.grid(axis='x', visible=False)  # Explicitly turn off vertical grid lines
    ax.grid(axis='y', alpha=0.7, linestyle=':', linewidth=0.8)  # Darker dotted horizontal lines
    ax.set_axisbelow(True)
    
    # Set y-axis to start from 0 and add some padding at top
    ax.set_ylim(0, max(counts) * 1.15)
    
    # Remove top and right spines for cleaner look
    sns.despine(top=True, right=True)
    
    # Add subtle background color
    ax.set_facecolor('#fafafa')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save with high quality for publications
    histogram_path = os.path.join(output_dir, "tool_call_distribution_histogram.png")
    plt.savefig(histogram_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none',
                format='png')
    
    # Also save as PDF for academic use
    pdf_path = os.path.join(output_dir, "tool_call_distribution_histogram.pdf")
    plt.savefig(pdf_path, bbox_inches='tight', 
                facecolor='white', edgecolor='none',
                format='pdf')
    
    print(f"Histogram saved to: {histogram_path}")
    print(f"PDF version saved to: {pdf_path}")
    
    # Show the plot
    plt.show()
    
    # Reset style to default
    sns.reset_defaults()
    
    return histogram_path

def create_donut_chart(tool_counter, output_dir, center_text="Strict Verification"):
    """
    Create and save donut chart visualization of tool call distribution.
    
    Args:
        tool_counter (Counter): Counter object with tool call counts
        output_dir (str): Directory to save outputs
        center_text (str): Text to display in the center of the donut chart
        
    Returns:
        str: Path to saved donut chart
    """
    if not tool_counter:
        print("No data to visualize")
        return None
    
    # Set seaborn style for academic publications
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.4)
    
    # Define consistent color mapping for each tool
    tool_color_map = {
        'TRELLIS': '#5ec962',  # Blue
        'SAM 2': '#3b528b',   
        'DAv2': '#21918c'      # Green
    }
    
    # Prepare data for plotting
    tools = list(tool_counter.keys())
    # Format tool names properly
    formatted_tools = []
    for tool in tools:
        if tool.lower() == 'trellis':
            formatted_tools.append('TRELLIS')
        elif tool.lower() == 'sam2':
            formatted_tools.append('SAM 2')
        elif tool.lower() == 'dav2':
            formatted_tools.append('DAv2')
        else:
            formatted_tools.append(tool)
    tools = formatted_tools
    counts = list(tool_counter.values())
    total_calls = sum(counts)
    
    # Calculate percentages
    percentages = [(count / total_calls) * 100 for count in counts]
    
    # Create figure with appropriate size for academic papers
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Assign consistent colors based on tool names
    colors = [tool_color_map.get(tool, '#808080') for tool in tools]
    
    # Create donut chart (pie chart with a hole)
    wedges, texts, autotexts = ax.pie(counts, labels=tools, autopct='%1.1f%%',
                                      colors=colors, startangle=90,
                                      pctdistance=0.75, labeldistance=1.1,
                                      textprops={'fontsize': 16, 'fontweight': 'bold'})
    
    # Create the donut hole
    centre_circle = plt.Circle((0, 0), 0.50, fc='white')
    fig.gca().add_artist(centre_circle)
    
    # Customize the text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(18)
        autotext.set_fontweight('bold')
    
    # Add title in the center
    ax.text(0, 0, center_text, 
            horizontalalignment='center', verticalalignment='center',
            fontsize=20, fontweight='bold', color='black')
    
    # Ensure the pie chart is circular
    ax.axis('equal')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save with high quality for publications
    donut_path = os.path.join(output_dir, "tool_call_distribution_donut.png")
    plt.savefig(donut_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none',
                format='png')
    
    # Also save as PDF for academic use
    pdf_path = os.path.join(output_dir, "tool_call_distribution_donut.pdf")
    plt.savefig(pdf_path, bbox_inches='tight', 
                facecolor='white', edgecolor='none',
                format='pdf')
    
    print(f"Donut chart saved to: {donut_path}")
    print(f"PDF version saved to: {pdf_path}")
    
    # Show the plot
    plt.show()
    
    # Reset style to default
    sns.reset_defaults()
    
    return donut_path

def save_detailed_results(tool_counter, processing_stats, output_dir):
    """
    Save detailed analysis results to JSON file.
    
    Args:
        tool_counter (Counter): Tool call counts
        processing_stats (dict): Processing statistics
        output_dir (str): Output directory
        
    Returns:
        str: Path to saved results file
    """
    # Calculate additional statistics
    total_calls = sum(tool_counter.values())
    unique_tools = len(tool_counter)
    
    # Create comprehensive results dictionary
    results_data = {
        "analysis_summary": {
            "total_tool_calls": total_calls,
            "unique_tools": unique_tools,
            "most_used_tool": tool_counter.most_common(1)[0] if tool_counter else None,
            "least_used_tool": tool_counter.most_common()[-1] if tool_counter else None
        },
        "tool_call_counts": dict(tool_counter.most_common()),
        "tool_call_percentages": {
            tool: round((count / total_calls) * 100, 2) if total_calls > 0 else 0
            for tool, count in tool_counter.items()
        },
        "processing_statistics": processing_stats,
        "metadata": {
            "script_version": "1.0",
            "analysis_type": "tool_call_distribution",
            "data_source": "complete_trace.json files"
        }
    }
    
    # Save to JSON file
    results_path = os.path.join(output_dir, "tool_call_analysis_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, sort_keys=True, ensure_ascii=False)
    
    print(f"Detailed results saved to: {results_path}")
    return results_path

def print_analysis_summary(tool_counter, processing_stats):
    """
    Print a comprehensive summary of the analysis results.
    
    Args:
        tool_counter (Counter): Tool call counts
        processing_stats (dict): Processing statistics
    """
    print("\n" + "="*60)
    print("TOOL CALL DISTRIBUTION ANALYSIS SUMMARY")
    print("="*60)
    
    # Processing statistics
    print(f"\nProcessing Statistics:")
    print(f"  • Total question folders found: {processing_stats['total_folders']}")
    print(f"  • Successfully processed files: {processing_stats['processed_files']}")
    print(f"  • Files with errors/missing: {processing_stats['error_files']}")
    print(f"  • Processing success rate: {processing_stats['success_rate']:.1f}%")
    
    if tool_counter:
        total_calls = sum(tool_counter.values())
        print(f"\nTool Call Analysis:")
        print(f"  • Total tool calls found: {total_calls}")
        print(f"  • Unique tools used: {len(tool_counter)}")
        print(f"  • Average calls per processed file: {total_calls / max(processing_stats['processed_files'], 1):.1f}")
        
        print(f"\nTool Usage Distribution:")
        for i, (tool, count) in enumerate(tool_counter.most_common(), 1):
            percentage = (count / total_calls) * 100
            print(f"  {i:2d}. {tool:12s}: {count:4d} calls ({percentage:5.1f}%)")
    else:
        print(f"\nNo tool calls found in any processed files!")
    
    print("\n" + "="*60)

def analyze_tool_calls():
    """
    Main function to analyze tool call distribution from LLM reasoning traces.
    
    This function processes all question subfolders in the CLEVR evaluation directory,
    extracts tool calls from complete_trace.json files, and generates comprehensive
    analysis with visualizations and detailed statistics.
    """
    # Configuration
    questions_dir = "/home/rmc/spatial_trace/spatial_trace/spatial_trace/evaluation/experiments/clevr_human_WITH_hard_verification_large/questions"
    output_dir = "/home/rmc/spatial_trace/spatial_trace/spatial_trace/"
    
    print("Starting Tool Call Distribution Analysis...")
    print(f"Questions directory: {questions_dir}")
    print(f"Output directory: {output_dir}")
    
    # Validate directories
    if not validate_directories(questions_dir, output_dir):
        print("Directory validation failed. Exiting.")
        return
    
    # Initialize data structures
    tool_counter = Counter()
    error_log = []
    
    # Find all question folders
    question_pattern = os.path.join(questions_dir, "q*")
    question_folders = sorted(glob.glob(question_pattern))
    
    if not question_folders:
        print(f"No question folders found matching pattern: {question_pattern}")
        return
    
    print(f"Found {len(question_folders)} question folders to process")
    
    # Process each question folder
    processed_files = 0
    error_files = 0
    trellis_folders = []  # Track folders where TRELLIS is called
    
    for folder in tqdm(question_folders, desc="Processing question folders", unit="folder"):
        folder_name = os.path.basename(folder)
        tool_calls, error = process_question_folder(folder)
        
        if error:
            error_files += 1
            error_log.append(f"{folder_name}: {error}")
        else:
            processed_files += 1
            # Add tool calls to counter
            folder_has_trellis = False
            for tool in tool_calls:
                tool_counter[tool] += 1
                # Check if TRELLIS is called in this folder
                if tool.lower() == 'trellis':
                    folder_has_trellis = True
            
            # Track folders with TRELLIS calls
            if folder_has_trellis:
                trellis_folders.append(folder_name)
    
    # Calculate processing statistics
    total_folders = len(question_folders)
    success_rate = (processed_files / total_folders) * 100 if total_folders > 0 else 0
    
    processing_stats = {
        "total_folders": total_folders,
        "processed_files": processed_files,
        "error_files": error_files,
        "success_rate": success_rate
    }
    
    # Print summary
    print_analysis_summary(tool_counter, processing_stats)
    
    # Print TRELLIS directories
    if trellis_folders:
        print(f"\n" + "="*60)
        print("DIRECTORIES WHERE TRELLIS IS CALLED")
        print("="*60)
        print(f"Found TRELLIS calls in {len(trellis_folders)} directories:")
        for i, folder in enumerate(sorted(trellis_folders), 1):
            print(f"  {i:3d}. {folder}")
        print("="*60)
    else:
        print(f"\nNo directories with TRELLIS calls found.")
    
    # Save error log if there were errors
    if error_log:
        error_log_path = os.path.join(output_dir, "tool_call_analysis_errors.log")
        with open(error_log_path, 'w', encoding='utf-8') as f:
            f.write("Tool Call Analysis Error Log\n")
            f.write("="*50 + "\n\n")
            for error in error_log:
                f.write(f"{error}\n")
        print(f"Error log saved to: {error_log_path}")
    
    # Generate outputs if we found tool calls
    if tool_counter:
        # Create visualizations
        histogram_path = create_visualization(tool_counter, output_dir)
        donut_path = create_donut_chart(tool_counter, output_dir, center_text=r"$\tau = 5$")
        
        # Save detailed results
        results_path = save_detailed_results(tool_counter, processing_stats, output_dir)
        
        print(f"\nAnalysis completed successfully!")
        print(f"Generated files:")
        if histogram_path:
            print(f"  • Histogram: {histogram_path}")
        if donut_path:
            print(f"  • Donut chart: {donut_path}")
        print(f"  • Results: {results_path}")
        if error_log:
            print(f"  • Error log: {error_log_path}")
    else:
        print(f"\nAnalysis completed, but no tool calls were found.")
        print(f"This might indicate:")
        print(f"  • Different JSON structure than expected")
        print(f"  • No tool calls in the processed traces")
        print(f"  • Issues with file access or parsing")
        
        if error_log:
            print(f"\nCheck the error log for details: {error_log_path}")

if __name__ == "__main__":
    try:
        analyze_tool_calls()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 