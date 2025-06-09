"""
Abstract base class for spatial reasoning tools.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Abstract base class for all spatial reasoning tools."""
    
    def __init__(self, name: str, conda_env: str, tool_path: Path):
        """
        Initialize the base tool.
        
        Args:
            name: Human-readable name of the tool
            conda_env: Name of the conda environment for this tool
            tool_path: Path to the tool's directory
        """
        self.name = name
        self.conda_env = conda_env
        self.tool_path = Path(tool_path)
        
    @abstractmethod
    def run(self, image_path: Path) -> Tuple[str, Optional[Path]]:
        """
        Run the tool on an input image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (status_message, output_image_path)
            - status_message: Description of the tool's output or error
            - output_image_path: Path to generated output image, or None if failed
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """
        Get a description of what this tool does.
        
        Returns:
            String description of the tool's functionality
        """
        pass
    
    def is_available(self) -> bool:
        """
        Check if the tool is properly configured and available.
        
        Returns:
            True if tool can be used, False otherwise
        """
        try:
            # Check if tool directory exists
            if not self.tool_path.exists():
                logger.warning(f"Tool path does not exist: {self.tool_path}")
                return False
            
            # Additional checks can be implemented by subclasses
            return self._check_availability()
        except Exception as e:
            logger.error(f"Error checking tool availability: {e}")
            return False
    
    def _check_availability(self) -> bool:
        """
        Subclasses can override this for additional availability checks.
        
        Returns:
            True if tool is available, False otherwise
        """
        return True
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get tool configuration information.
        
        Returns:
            Dictionary containing tool configuration
        """
        return {
            "name": self.name,
            "conda_env": self.conda_env,
            "tool_path": str(self.tool_path),
            "available": self.is_available()
        } 