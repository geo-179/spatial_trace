"""
Image utility functions for spatial reasoning.
"""
import base64
import mimetypes
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def encode_image_to_base64(image_path: Path) -> Optional[str]:
    """
    Reads an image file and encodes it as a Base64 string.

    Args:
        image_path: Path to the image file.

    Returns:
        Base64 encoded string with the correct data URI prefix, or None if failed.
    """
    try:
        # Convert to Path object if it's a string
        image_path = Path(image_path)
        
        # Check if file exists
        if not image_path.exists():
            logger.error(f"Image file not found at {image_path}")
            return None
        
        # Guess the MIME type of the image
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type is None:
            # Default to jpeg if the type can't be determined
            mime_type = "image/jpeg"
            logger.warning(f"Could not determine MIME type for {image_path}, defaulting to image/jpeg")
        
        # Read and encode the image
        with open(image_path, "rb") as image_file:
            encoded_bytes = base64.b64encode(image_file.read())
            encoded_string = encoded_bytes.decode('utf-8')
            
        data_uri = f"data:{mime_type};base64,{encoded_string}"
        logger.debug(f"Successfully encoded image {image_path} to base64 (MIME: {mime_type})")
        return data_uri
        
    except FileNotFoundError:
        logger.error(f"Image file not found at {image_path}")
        return None
    except PermissionError:
        logger.error(f"Permission denied reading image file at {image_path}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error encoding image {image_path}: {e}")
        return None


def validate_image_path(image_path: Path) -> bool:
    """
    Validate that an image path exists and is a valid image file.
    
    Args:
        image_path: Path to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        image_path = Path(image_path)
        
        # Check if file exists
        if not image_path.exists():
            logger.warning(f"Image path does not exist: {image_path}")
            return False
        
        # Check if it's a file
        if not image_path.is_file():
            logger.warning(f"Image path is not a file: {image_path}")
            return False
        
        # Check file extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        if image_path.suffix.lower() not in valid_extensions:
            logger.warning(f"Image file has unsupported extension: {image_path.suffix}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating image path {image_path}: {e}")
        return False


def get_image_info(image_path: Path) -> dict:
    """
    Get basic information about an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with image information
    """
    try:
        image_path = Path(image_path)
        
        if not validate_image_path(image_path):
            return {"valid": False, "error": "Invalid image path"}
        
        # Get file stats
        stat = image_path.stat()
        mime_type, _ = mimetypes.guess_type(str(image_path))
        
        return {
            "valid": True,
            "path": str(image_path),
            "name": image_path.name,
            "size_bytes": stat.st_size,
            "mime_type": mime_type,
            "extension": image_path.suffix.lower()
        }
        
    except Exception as e:
        logger.error(f"Error getting image info for {image_path}: {e}")
        return {"valid": False, "error": str(e)} 