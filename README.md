# Spatial Trace

A framework for generating spatial reasoning traces using Large Language Models (LLMs) and computer vision tools like Depth-Anything-V2 (DAV2) and Segment Anything Model 2 (SAM2).

## Overview

Spatial Trace enables AI systems to perform step-by-step spatial reasoning by:
- Processing images with computer vision tools (segmentation, depth estimation)
- Using LLMs to generate reasoning traces
- Combining visual outputs with textual reasoning steps
- Producing interpretable spatial reasoning datasets

## Features

- **Tool Integration**: Seamless integration with SAM2 for segmentation and DAV2 for depth estimation
- **LLM Interface**: Structured communication with OpenAI's GPT models
- **Reasoning Traces**: Generation of step-by-step spatial reasoning processes
- **Modular Design**: Clean separation of concerns across tools, LLM interface, and data processing
- **Extensible Architecture**: Easy to add new tools and reasoning capabilities

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/spatial-trace.git
cd spatial-trace
```

### 2. Install the Package

#### Option A: Development Installation (Recommended)
```bash
# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

#### Option B: Standard Installation
```bash
pip install .
```

### 3. Set Up External Tools

#### SAM2 Setup
```bash
# Navigate to your SAM2 directory
cd /path/to/sam2
# Create conda environment for SAM2
conda create -n sam2 python=3.9
conda activate sam2
# Install SAM2 dependencies (follow SAM2 installation guide)
```

#### Depth-Anything-V2 Setup
```bash
# Navigate to your Depth-Anything-V2 directory
cd /path/to/Depth-Anything-V2
# Create conda environment for DAV2
conda create -n DAv2 python=3.9
conda activate DAv2
# Install DAV2 dependencies (follow DAV2 installation guide)
```

### 4. Environment Configuration

Create a `.env` file in the project root:
```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Tool Paths (adjust these to your actual paths)
SAM2_PATH=/path/to/sam2
DAV2_PATH=/path/to/Depth-Anything-V2

# Conda Environment Names
SAM2_ENV=sam2
DAV2_ENV=DAv2
```

## Quick Start

### Basic Usage

```python
from spatial_trace.inference.pipeline import SpatialReasoningPipeline
from spatial_trace.llm_interface.openai_client import OpenAIClient

# Initialize the pipeline
llm_client = OpenAIClient()
pipeline = SpatialReasoningPipeline(llm_client)

# Run spatial reasoning
question = "Is the red block to the left of the blue sphere?"
image_path = "path/to/your/image.jpg"

trace = pipeline.generate_reasoning_trace(question, image_path)
print("Reasoning trace completed!")
```

### CLI Usage

```bash
# Run spatial reasoning pipeline
spatial-trace reason --question "Is the red block to the left of the blue sphere?" --image path/to/image.jpg

# Process a dataset
spatial-trace batch --config configs/experiments/run_config_001.yaml
```

## Project Structure

```
spatial_trace/
├── __init__.py
├── inference/              # Main reasoning pipeline
│   ├── __init__.py
│   ├── pipeline.py        # Core reasoning pipeline
│   └── trace_processor.py # Post-processing of traces
├── llm_interface/         # LLM communication
│   ├── __init__.py
│   ├── base_client.py    # Abstract LLM client
│   ├── openai_client.py  # OpenAI implementation
│   ├── output_parser.py  # Parse LLM responses
│   └── prompt_manager.py # System prompts and templates
├── tools/                 # Computer vision tools
│   ├── __init__.py
│   ├── base_tool.py      # Abstract tool interface
│   ├── sam2_tool.py      # SAM2 segmentation
│   ├── dav2_tool.py      # Depth-Anything-V2
│   └── tool_registry.py  # Tool management
├── utils/                 # Utilities
│   ├── __init__.py
│   ├── image_utils.py    # Image processing
│   ├── file_io.py        # File operations
│   └── logger.py         # Logging configuration
└── cli.py                # Command-line interface
```

## Configuration

System prompts and configuration files are stored in:
- `configs/prompts/` - Prompt templates and instructions
- `configs/experiments/` - Experiment configurations
- `configs/llm_models.yaml` - LLM model settings

## Development

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black spatial_trace/
isort spatial_trace/
```

### Type Checking
```bash
mypy spatial_trace/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{spatial_trace,
  title={Spatial Trace: A Framework for LLM-based Spatial Reasoning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/spatial-trace}
}
```