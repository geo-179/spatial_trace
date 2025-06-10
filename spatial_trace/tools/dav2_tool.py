"""
Depth-Anything-V2 (DAV2) tool for depth estimation.
"""
import subprocess
import logging
from pathlib import Path
from typing import Tuple, Optional

from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class DAV2Tool(BaseTool):
    """Depth-Anything-V2 tool for depth estimation."""

    def __init__(self, tool_path: Path, conda_env: str = "DAv2"):
        """
        Initialize DAV2 tool.

        Args:
            tool_path: Path to the Depth-Anything-V2 directory
            conda_env: Name of the conda environment for DAV2
        """
        super().__init__(name="DAV2", conda_env=conda_env, tool_path=tool_path)
        self.script_name = "depth_estimation.py"
        self.output_filename = "depth_colored.png"

    def run(self, image_path: Path) -> Tuple[str, Optional[Path]]:
        """
        Run DepthAnythingV2 to estimate depth of an image.

        Args:
            image_path: Path to the input image

        Returns:
            Tuple of (status_message, output_image_path)
        """
        logger.info(f"Running DepthAnythingV2 depth estimation on {image_path}")

        # Ensure paths are absolute for robust execution
        absolute_image_path = image_path.resolve()
        script_path = (self.tool_path / self.script_name).resolve()

        # CORRECTED COMMAND: The image path is a positional argument, not a named one.
        command = [
            "conda", "run", "-n", self.conda_env,
            "python", str(script_path),
            str(absolute_image_path),      # The image path is passed directly.
            "--outdir", str(self.tool_path) # The output directory is a named argument.
        ]

        logger.debug(f"Executing command: {' '.join(command)}")
        logger.debug(f"Working directory: {self.tool_path}")

        try:
            # Execute the command
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd=self.tool_path
            )
            logger.debug(f"DAV2 stdout: {result.stdout}")

            # Check for output file
            output_file_path = self.tool_path / self.output_filename

            if output_file_path.is_file():
                success_msg = f"Image with depth map at {output_file_path}"
                logger.info(f"DAV2 depth estimation successful: {output_file_path}")
                return success_msg, output_file_path
            else:
                error_msg = "DAV2 script did not produce the expected output file"
                logger.error(f"{error_msg}. Looked for: {output_file_path}")
                return error_msg, None

        except FileNotFoundError:
            error_msg = "Conda command not found. Make sure Conda is installed and in your PATH"
            logger.error(error_msg)
            return error_msg, None
        except subprocess.CalledProcessError as e:
            error_msg = f"DAV2 script failed: {e.stderr}"
            logger.error(f"DAV2 execution failed. Return code: {e.returncode}")
            logger.error(f"Stderr: {e.stderr}")
            return error_msg, None
        except Exception as e:
            error_msg = f"Unexpected error running DAV2: {str(e)}"
            logger.error(error_msg)
            return error_msg, None

    def get_description(self) -> str:
        """Get description of DAV2 tool."""
        return "A depth estimation tool that analyzes the relative distances of objects from the camera using Depth-Anything-V2."

    def _check_availability(self) -> bool:
        """Check if DAV2 script exists."""
        script_path = self.tool_path / self.script_name
        if not script_path.exists():
            logger.warning(f"DAV2 script not found: {script_path}")
            return False
        return True
