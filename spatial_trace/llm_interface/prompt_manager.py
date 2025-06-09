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
        self.system_prompts["spatial_reasoning"] = """
You are an expert AI in spatial reasoning. Your goal is to solve a user's question about an image by generating a step-by-step reasoning trace.

You have access to a suite of tools:
1. `sam2`: A segmentation tool. Call this to identify and outline objects in the image.
2. `dav2`: A depth estimation tool. Call this to understand the relative distances of objects from the camera.

At each step, your response MUST be a single, valid JSON object and nothing else. Do not add any explanatory text outside of the JSON structure.

Choose ONE of the following three actions inside the JSON object:

1. To call a tool:
   {"action": "tool_call", "tool_name": "tool_name_here"}

2. To reason without a tool:
   {"action": "reasoning", "text": "your_reasoning_step_here"}

3. To give the final answer (either "Yes/No", a number, or a short phrase):
   {"action": "final_answer", "text": "your_final_answer_here"}
"""
    
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
            # Include both text and image
            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Tool output: {tool_output_message}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": new_base64_image,
                            "detail": "low"  # Use low detail to save tokens
                        }
                    }
                ]
            }
        else:
            # Text only
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
        # Implementation depends on desired file format
        # For now, this is a placeholder for future enhancement
        logger.info(f"Loading prompts from {file_path} (not yet implemented)")


# Global prompt manager instance
prompt_manager = PromptManager() 