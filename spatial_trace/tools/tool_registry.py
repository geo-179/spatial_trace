"""
Tool registry for managing and dispatching spatial reasoning tools.
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Type
import os

from .base_tool import BaseTool
from .sam2_tool import SAM2Tool
from .dav2_tool import DAV2Tool
from .trellis_tool import TrellisTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing spatial reasoning tools."""

    def __init__(self):
        """Initialize the tool registry."""
        self.tools: Dict[str, BaseTool] = {}
        self._tool_classes: Dict[str, Type[BaseTool]] = {
            "sam2": SAM2Tool,
            "dav2": DAV2Tool,
            "trellis": TrellisTool,
        }

    def register_tool(self, tool_name: str, tool: BaseTool) -> None:
        """
        Register a tool instance.

        Args:
            tool_name: Name to register the tool under
            tool: Tool instance to register
        """
        self.tools[tool_name.lower()] = tool
        logger.info(f"Registered tool: {tool_name}")

    def configure_from_env(self) -> None:
        """
        Configure tools from environment variables.
        Expected environment variables:
        - SAM2_PATH: Path to SAM2 directory
        - DAV2_PATH: Path to Depth-Anything-V2 directory
        - TRELLIS_PATH: Path to TRELLIS directory
        - SAM2_ENV: Conda environment name for SAM2 (default: sam2)
        - DAV2_ENV: Conda environment name for DAV2 (default: DAv2)
        - TRELLIS_ENV: Conda environment name for Trellis (default: trellis)
        """
        sam2_path = os.getenv("SAM2_PATH")
        if sam2_path:
            sam2_env = os.getenv("SAM2_ENV", "sam2")
            try:
                sam2_tool = SAM2Tool(Path(sam2_path), sam2_env)
                self.register_tool("sam2", sam2_tool)
            except Exception as e:
                logger.error(f"Failed to configure SAM2 tool: {e}")
        else:
            logger.warning("SAM2_PATH not set in environment variables")

        dav2_path = os.getenv("DAV2_PATH")
        if dav2_path:
            dav2_env = os.getenv("DAV2_ENV", "DAv2")
            try:
                dav2_tool = DAV2Tool(Path(dav2_path), dav2_env)
                self.register_tool("dav2", dav2_tool)
            except Exception as e:
                logger.error(f"Failed to configure DAV2 tool: {e}")
        else:
            logger.warning("DAV2_PATH not set in environment variables")

        trellis_path = os.getenv("TRELLIS_PATH")
        if trellis_path:
            trellis_env = os.getenv("TRELLIS_ENV", "trellis")
            try:
                trellis_tool = TrellisTool(Path(trellis_path), trellis_env)
                self.register_tool("trellis", trellis_tool)
            except Exception as e:
                logger.error(f"Failed to configure Trellis tool: {e}")
        else:
            logger.warning("TRELLIS_PATH not set in environment variables")

    def configure_tool(self, tool_name: str, tool_path: Path, conda_env: str) -> bool:
        """
        Configure a specific tool.

        Args:
            tool_name: Name of the tool to configure
            tool_path: Path to the tool's directory
            conda_env: Conda environment name

        Returns:
            True if configuration successful, False otherwise
        """
        tool_name_lower = tool_name.lower()

        if tool_name_lower not in self._tool_classes:
            logger.error(f"Unknown tool: {tool_name}")
            return False

        try:
            tool_class = self._tool_classes[tool_name_lower]
            tool = tool_class(tool_path, conda_env)
            self.register_tool(tool_name_lower, tool)
            return True
        except Exception as e:
            logger.error(f"Failed to configure {tool_name}: {e}")
            return False

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(tool_name.lower())

    def run_tool(self, tool_name: str, image_path: Path) -> Tuple[str, Optional[Path]]:
        """
        Run a tool and return status message and output image path.

        Args:
            tool_name: Name of the tool to run
            image_path: Path to the input image

        Returns:
            Tuple of (status_message, output_image_path)
            - output_image_path: Actual path where tool saved its output
        """
        if tool_name not in self.tools:
            error_msg = f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
            logger.error(error_msg)
            return error_msg, None

        tool = self.tools[tool_name]

        if not tool.is_available():
            error_msg = f"Tool '{tool_name}' is not available"
            logger.error(error_msg)
            return error_msg, None

        try:
            logger.info(f"Running tool '{tool_name}' on {image_path}")
            status_msg, output_path = tool.run(image_path)

            if output_path:
                logger.info(f"Tool '{tool_name}' output saved to: {output_path}")

            return status_msg, output_path
        except Exception as e:
            error_msg = f"Error running tool '{tool_name}': {e}"
            logger.error(error_msg)
            return error_msg, None

    def list_tools(self) -> List[str]:
        """
        Get list of registered tool names.

        Returns:
            List of tool names
        """
        return list(self.tools.keys())

    def get_tool_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions for all registered tools.

        Returns:
            Dictionary mapping tool names to descriptions
        """
        descriptions = {}
        for name, tool in self.tools.items():
            descriptions[name] = tool.get_description()
        return descriptions

    def check_tool_availability(self) -> Dict[str, bool]:
        """
        Check availability of all registered tools.

        Returns:
            Dictionary mapping tool names to availability status
        """
        availability = {}
        for name, tool in self.tools.items():
            availability[name] = tool.is_available()
        return availability


tool_registry = ToolRegistry()
