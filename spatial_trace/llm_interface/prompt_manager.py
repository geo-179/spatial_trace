"""
Prompt manager for handling system prompts and templates.
"""
import logging
from typing import Dict, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class PromptManager:
    """Manager for system prompts and prompt templates."""

    def __init__(self):
        """Initialize the prompt manager."""
        self.system_prompts = {}
        self._load_default_prompts()

    def _load_default_prompts(self) -> None:
        """Load default system prompts."""
        prompt_file_path = Path(__file__).parent.parent.parent / "configs" / "prompts" / "spatial_reasoning_system.txt"
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                self.system_prompts["spatial_reasoning"] = f.read().strip()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_file_path}")
            
        verifier_prompt_file_path = Path(__file__).parent.parent.parent / "configs" / "prompts" / "verifier_system.txt"
        try:
            with open(verifier_prompt_file_path, 'r', encoding='utf-8') as f:
                self.system_prompts["verifier"] = f.read().strip()
        except FileNotFoundError:
            logger.error(f"Verifier prompt file not found: {verifier_prompt_file_path}")

    def get_system_prompt(self, prompt_name: str) -> str:
        """
        Get a system prompt by name.

        Args:
            prompt_name: Name of the prompt to retrieve

        Returns:
            System prompt text

        Raises:
            KeyError: If prompt not found
        """
        if prompt_name not in self.system_prompts:
            available_prompts = list(self.system_prompts.keys())
            raise KeyError(f"Prompt '{prompt_name}' not found. Available prompts: {available_prompts}")

        return self.system_prompts[prompt_name]

    def add_system_prompt(self, prompt_name: str, prompt_text: str) -> None:
        """
        Add or update a system prompt.

        Args:
            prompt_name: Name of the prompt
            prompt_text: Text content of the prompt
        """
        self.system_prompts[prompt_name] = prompt_text
        logger.info(f"Added/updated system prompt: {prompt_name}")

    def list_prompts(self) -> List[str]:
        """
        Get list of available prompt names.

        Returns:
            List of prompt names
        """
        return list(self.system_prompts.keys())

    def create_initial_messages(
        self,
        question: str,
        base64_image: str,
        prompt_name: str = "spatial_reasoning"
    ) -> List[Dict[str, Any]]:
        """
        Create initial message structure for spatial reasoning.

        Args:
            question: The spatial reasoning question
            base64_image: Base64 encoded image
            prompt_name: Name of the system prompt to use

        Returns:
            List of message dictionaries ready for LLM API
        """
        system_prompt = self.get_system_prompt(prompt_name)

        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Solve this spatial reasoning question: {question}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image,
                            "detail": "high"
                        }
                    }
                ]
            }
        ]

    def create_tool_result_message(
        self,
        tool_output_message: str,
        new_base64_image: str = None
    ) -> Dict[str, Any]:
        """
        Create a message for tool results.

        Args:
            tool_output_message: Text description of tool output
            new_base64_image: Base64 encoded image from tool (optional)

        Returns:
            Message dictionary for tool results
        """
        if new_base64_image:
            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Tool output: {tool_output_message}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": new_base64_image,
                            "detail": "low"
                        }
                    }
                ]
            }
        else:
            return {
                "role": "user",
                "content": f"Tool output: {tool_output_message}"
            }

    def load_prompts_from_file(self, file_path: Path) -> None:
        """
        Load prompts from a configuration file.

        Args:
            file_path: Path to the prompt configuration file

        Note:
            This method can be extended to support YAML, JSON, or other formats
        """
        logger.info(f"Loading prompts from {file_path} (not yet implemented)")


prompt_manager = PromptManager()
