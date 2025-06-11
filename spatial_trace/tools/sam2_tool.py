"""
SAM2 (Segment Anything Model 2) tool for image segmentation.
"""
import subprocess
import logging
from pathlib import Path
from typing import Tuple, Optional

from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class SAM2Tool(BaseTool):
    """SAM2 tool for image segmentation."""

    def __init__(self, tool_path: Path, conda_env: str = "sam2"):
        """
        Initialize SAM2 tool.

        Args:
            tool_path: Path to the SAM2 directory
            conda_env: Name of the conda environment for SAM2
        """
        super().__init__(name="SAM2", conda_env=conda_env, tool_path=tool_path)
        self.script_name = "segment_image.py"
        self.output_filename = "segmented_image.png"

    def run(self, image_path: Path) -> Tuple[str, Optional[Path]]:
        """
        Run SAM2 to segment an image.

        Args:
            image_path: Path to the input image

        Returns:
            Tuple of (status_message, output_image_path)
        """
        logger.info(f"Running SAM2 segmentation on {image_path}")

        absolute_image_path = image_path.resolve()
        script_path = (self.tool_path / self.script_name).resolve()

        command = [
            "conda", "run", "-n", self.conda_env,
            "python", str(script_path),
            str(absolute_image_path)
        ]

        logger.debug(f"Executing command: {' '.join(command)}")
        logger.debug(f"Working directory: {self.tool_path}")

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd=self.tool_path
            )
            logger.debug(f"SAM2 stdout: {result.stdout}")

            output_file_path = self.tool_path / self.output_filename

            if output_file_path.is_file():
                success_msg = f"Image with segmentation mask at {output_file_path}"
                logger.info(f"SAM2 segmentation successful: {output_file_path}")
                return success_msg, output_file_path
            else:
                error_msg = "SAM2 script did not produce the expected output file"
                logger.error(f"{error_msg}. Looked for: {output_file_path}")
                return error_msg, None

        except FileNotFoundError:
            error_msg = "Conda command not found. Make sure Conda is installed and in your PATH"
            logger.error(error_msg)
            return error_msg, None
        except subprocess.CalledProcessError as e:
            error_msg = f"SAM2 script failed: {e.stderr}"
            logger.error(f"SAM2 execution failed. Return code: {e.returncode}")
            logger.error(f"Stderr: {e.stderr}")
            return error_msg, None
        except Exception as e:
            error_msg = f"Unexpected error running SAM2: {str(e)}"
            logger.error(error_msg)
            return error_msg, None

    def get_description(self) -> str:
        """Get description of SAM2 tool."""
        return "A segmentation tool that identifies and outlines objects in images using the Segment Anything Model 2."

    def _check_availability(self) -> bool:
        """Check if SAM2 script exists."""
        script_path = self.tool_path / self.script_name
        if not script_path.exists():
            logger.warning(f"SAM2 script not found: {script_path}")
            return False
        return True
