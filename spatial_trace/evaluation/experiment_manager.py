"""
Utilities for managing experiment organization and structure.
"""
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ExperimentManager:
    """Manages experiment directory structure and organization."""

    def __init__(self, base_experiments_dir: Path = None):
        """Initialize experiment manager."""
        # Default to local experiments directory in evaluation folder
        if base_experiments_dir is None:
            base_experiments_dir = Path(__file__).parent / "experiments"

        self.base_dir = base_experiments_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list_experiments(self) -> List[str]:
        """List all available experiments."""
        experiments = []
        for exp_dir in self.base_dir.iterdir():
            if exp_dir.is_dir() and (exp_dir / "summary" / "experiment_config.json").exists():
                experiments.append(exp_dir.name)
        return sorted(experiments)

    def get_experiment_info(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific experiment."""
        exp_dir = self.base_dir / experiment_name
        config_file = exp_dir / "summary" / "experiment_config.json"

        if not config_file.exists():
            return None

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            # Add directory structure info
            questions_dir = exp_dir / "questions"
            if questions_dir.exists():
                question_dirs = [d.name for d in questions_dir.iterdir() if d.is_dir()]
                config["total_questions"] = len(question_dirs)
                config["question_directories"] = question_dirs[:5]  # First 5 for preview

            return config
        except Exception as e:
            logger.error(f"Error reading experiment info for {experiment_name}: {e}")
            return None

    def cleanup_experiment(self, experiment_name: str) -> bool:
        """Remove an experiment directory."""
        exp_dir = self.base_dir / experiment_name
        if exp_dir.exists():
            try:
                shutil.rmtree(exp_dir)
                logger.info(f"Removed experiment: {experiment_name}")
                return True
            except Exception as e:
                logger.error(f"Error removing experiment {experiment_name}: {e}")
                return False
        return False

    def archive_experiment(self, experiment_name: str) -> bool:
        """Archive an experiment by moving it to an archive subdirectory."""
        exp_dir = self.base_dir / experiment_name
        archive_dir = self.base_dir / "archived"
        archive_dir.mkdir(exist_ok=True)

        if exp_dir.exists():
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archived_name = f"{experiment_name}_{timestamp}"
                shutil.move(str(exp_dir), str(archive_dir / archived_name))
                logger.info(f"Archived experiment {experiment_name} as {archived_name}")
                return True
            except Exception as e:
                logger.error(f"Error archiving experiment {experiment_name}: {e}")
                return False
        return False


if __name__ == "__main__":
    manager = ExperimentManager()

    print("Available experiments:")
    for exp in manager.list_experiments():
        info = manager.get_experiment_info(exp)
        if info:
            print(f"  {exp}: {info.get('total_questions', 0)} questions")
