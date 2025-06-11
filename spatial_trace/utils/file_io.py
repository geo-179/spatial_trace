"""
File I/O utilities for spatial reasoning.
"""
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


def read_csv_data(file_path: Path) -> Optional[pd.DataFrame]:
    """
    Reads a CSV file into a pandas DataFrame.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the data, or None if an error occurs
    """
    try:
        file_path = Path(file_path)
        logger.info(f"Reading CSV data from: {file_path}")
        
        if not file_path.exists():
            logger.error(f"CSV file not found: {file_path}")
            return None
        
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        return df
        
    except FileNotFoundError:
        logger.error(f"The file was not found at the specified path: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"The CSV file is empty: {file_path}")
        return None
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading CSV file {file_path}: {e}")
        return None


def save_json_data(data: Dict[Any, Any], file_path: Path, indent: int = 2) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path where to save the JSON file
        indent: JSON indentation for pretty printing
        
    Returns:
        True if successful, False otherwise
    """
    try:
        file_path = Path(file_path)
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        
        logger.info(f"Successfully saved JSON data to: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving JSON data to {file_path}: {e}")
        return False


def load_json_data(file_path: Path) -> Optional[Dict[Any, Any]]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data or None if error
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"JSON file not found: {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Successfully loaded JSON data from: {file_path}")
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return None


def save_reasoning_trace(trace: List[Dict[str, Any]], output_path: Path, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Save a reasoning trace to a JSON file with metadata.
    
    Args:
        trace: List of reasoning steps
        output_path: Path where to save the trace
        metadata: Optional metadata to include
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_data = {
            "metadata": metadata or {},
            "trace": trace,
            "trace_length": len(trace)
        }
        
        return save_json_data(output_data, output_path)
        
    except Exception as e:
        logger.error(f"Error saving reasoning trace to {output_path}: {e}")
        return False


def load_reasoning_trace(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load a reasoning trace from a JSON file.
    
    Args:
        file_path: Path to the trace file
        
    Returns:
        Dictionary containing trace data or None if error
    """
    return load_json_data(file_path)


def ensure_directory_exists(directory_path: Path) -> bool:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        directory_path = Path(directory_path)
        directory_path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory_path}: {e}")
        return False


def get_file_info(file_path: Path) -> Dict[str, Any]:
    """
    Get information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"exists": False, "error": "File not found"}
        
        stat = file_path.stat()
        
        return {
            "exists": True,
            "path": str(file_path),
            "name": file_path.name,
            "size_bytes": stat.st_size,
            "is_file": file_path.is_file(),
            "is_directory": file_path.is_dir(),
            "suffix": file_path.suffix
        }
        
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {e}")
        return {"exists": False, "error": str(e)} 