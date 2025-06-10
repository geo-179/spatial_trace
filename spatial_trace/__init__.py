"""
Spatial Trace: A framework for LLM-based spatial reasoning.

This package provides tools and infrastructure for generating step-by-step
spatial reasoning traces using Large Language Models and computer vision tools.
"""

__version__ = "0.1.0"

# Main components
from .inference import SpatialReasoningPipeline, TraceProcessor, TraceVerifier, VerificationResult
from .llm_interface import OpenAIClient, BaseLLMClient
from .tools import tool_registry, SAM2Tool, DAV2Tool
from .utils import get_logger

# For convenience, expose the main pipeline
def create_pipeline(**kwargs):
    """Create a spatial reasoning pipeline with default settings."""
    return SpatialReasoningPipeline(**kwargs)

__all__ = [
    # Main pipeline
    "SpatialReasoningPipeline",
    "TraceProcessor",
    "TraceVerifier",
    "VerificationResult",
    "create_pipeline",

    # LLM clients
    "OpenAIClient",
    "BaseLLMClient",

    # Tools
    "tool_registry",
    "SAM2Tool",
    "DAV2Tool",

    # Utilities
    "get_logger",

    # Package info
    "__version__"
]
