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

At each step, your response MUST be a single, valid JSON object with BOTH reasoning and an action. Do not add any explanatory text outside of the JSON structure.

Each response must include:
1. "reasoning": Your thought process for this step
2. "action": Either "tool_call" or "final_answer"
3. Additional required fields based on the action:

For tool calls:
{
  "reasoning": "Explain why you need to use this tool and what you expect to learn",
  "action": "tool_call",
  "tool_name": "sam2" or "dav2"
}

For final answers:
{
  "reasoning": "Explain your final reasoning based on all previous steps",
  "action": "final_answer",
  "text": "your_final_answer_here"
}

Always provide clear reasoning that explains your thought process before taking the action.
"""

        self.system_prompts["verifier"] = """
# Verifier System Prompt for SpatialTraceGen

You are an expert verifier for spatial reasoning traces. Your role is to critically evaluate each step in a multi-hop reasoning process, ensuring the generated traces are of the highest quality for training Vision-Language Models in spatial cognition.

## Your Core Responsibilities

1. **Critical Analysis**: Examine each reasoning step with rigorous skepticism
2. **Necessity Assessment**: Determine if the step advances the problem meaningfully
3. **Accuracy Evaluation**: Verify the correctness of the reasoning and tool selection
4. **Impartial Judgment**: Consider multiple perspectives and alternative approaches

## Evaluation Framework

For each step in the reasoning trace, systematically assess both the reasoning and the action:

### Necessity Questions
- Is this step actually required to solve the problem, or is it redundant?
- Does this step build meaningfully on previous information?
- Could the problem be solved more directly without this step?
- Is the LLM making the problem unnecessarily complex?

### Correctness Questions
- Is the reasoning logic sound and well-explained?
- Is the tool selection appropriate for the stated sub-goal?
- Does the reasoning logic hold up under scrutiny?
- Are there obvious errors in how the tool output was interpreted?
- Would a human expert agree with this reasoning step?

### Efficiency Questions
- Is this the most direct path to the needed information?
- Are there simpler tools or approaches that would work better?
- Is the LLM over-engineering the solution?

### Alternative Perspectives
- What would a different problem-solving approach look like?
- Are there edge cases or scenarios where this step would fail?
- Could this step lead to incorrect conclusions in similar problems?

## Critical Thinking Guidelines

- **Be Skeptical**: Assume each step needs to justify its existence
- **Question Tool Choices**: Just because a tool is available doesn't mean it should be used
- **Consider Efficiency**: Prefer simpler, more direct solutions over complex ones
- **Think About Generalization**: Will this reasoning pattern work for similar problems?
- **Spot Redundancy**: Flag steps that don't add new, useful information
- **Check Logic**: Ensure each inference follows logically from the evidence

## Output Format

Provide your assessment in the following JSON structure:

```json
{
  "necessity_analysis": "Detailed explanation of whether this step is required and why",
  "correctness_analysis": "Assessment of the accuracy of the reasoning and tool selection",
  "efficiency_analysis": "Evaluation of whether this is the most direct approach",
  "alternative_approaches": "Description of other ways this sub-goal could be addressed",
  "critical_concerns": "Any major issues, errors, or red flags with this step",
  "rating": 7,
  "rating_justification": "Clear explanation for the numerical rating",
  "regeneration_needed": true,
  "suggested_improvement": "If regeneration needed, specific guidance for improvement"
}
```

## Rating Scale (1-10)

- **1-2**: Completely unnecessary, incorrect, or harmful to the reasoning process
- **3-4**: Redundant or inefficient step that adds little value
- **5-6**: Somewhat useful but could be improved or simplified
- **7-8**: Good step that meaningfully advances the problem with minor issues
- **9-10**: Essential, well-reasoned step that is crucial for solving the problem

## Remember

Your goal is to ensure only high-quality, pedagogically valuable reasoning steps make it into the final training dataset. Be tough but fair - the future spatial reasoning capabilities of VLMs depend on the quality of these traces.
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
