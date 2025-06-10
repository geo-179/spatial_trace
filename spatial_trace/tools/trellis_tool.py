"""
Trellis tool for generating top-down view of 3D scenes.
"""
import subprocess
import logging
from pathlib import Path
from typing import Tuple, Optional

from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class TrellisTool(BaseTool):
    """Trellis tool for generating top-down view of 3D scenes."""

    def __init__(self, tool_path: Path, conda_env: str = "trellis"):
        """
        Initialize Trellis tool.

        Args:
            tool_path: Path to the TRELLIS directory
            conda_env: Name of the conda environment for Trellis
        """
        super().__init__(name="Trellis", conda_env=conda_env, tool_path=tool_path)
        self.script_name = "generate_topdown.py"
        self.output_filename = "novel_view3_topdown.png"

    def run(self, image_path: Path) -> Tuple[str, Optional[Path]]:
        """
        Run Trellis to generate top-down view of an image.

        Args:
            image_path: Path to the input image

        Returns:
            Tuple of (status_message, output_image_path)
        """
        logger.info(f"Running Trellis top-down view generation on {image_path}")

        # Ensure paths are absolute for robust execution
        absolute_image_path = image_path.resolve()
        script_path = (self.tool_path / self.script_name).resolve()

        # Prepare the command with absolute paths
        command = [
            "conda", "run", "-n", self.conda_env,
            "python", str(script_path),
            str(absolute_image_path)
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
            logger.debug(f"Trellis stdout: {result.stdout}")

            # Check for output file
            output_file_path = self.tool_path / self.output_filename

            if output_file_path.is_file():
                success_msg = f"Top-down view image generated at {output_file_path}"
                logger.info(f"Trellis top-down view generation successful: {output_file_path}")
                return success_msg, output_file_path
            else:
                error_msg = "Trellis script did not produce the expected output file"
                logger.error(f"{error_msg}. Looked for: {output_file_path}")
                return error_msg, None

        except FileNotFoundError:
            error_msg = "Conda command not found. Make sure Conda is installed and in your PATH"
            logger.error(error_msg)
            return error_msg, None
        except subprocess.CalledProcessError as e:
            error_msg = f"Trellis script failed: {e.stderr}"
            logger.error(f"Trellis execution failed. Return code: {e.returncode}")
            logger.error(f"Stderr: {e.stderr}")
            return error_msg, None
        except Exception as e:
            error_msg = f"Unexpected error running Trellis: {str(e)}"
            logger.error(error_msg)
            return error_msg, None

    def get_description(self) -> str:
        """Get description of Trellis tool."""
        return "A 3D scene reconstruction tool that generates a top-down view of the scene using TRELLIS image-to-3D pipeline."

    def _check_availability(self) -> bool:
        """Check if Trellis script exists."""
        script_path = self.tool_path / self.script_name
        if not script_path.exists():
            logger.warning(f"Trellis script not found: {script_path}")
            return False
        return True
