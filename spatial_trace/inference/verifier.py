"""
Trace verifier for evaluating reasoning steps and triggering regeneration.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..llm_interface import OpenAIClient, prompt_manager
from ..utils import encode_image_to_base64

logger = logging.getLogger(__name__)


class VerificationResult:
    """Container for verification results."""

    def __init__(self, verification_data: Dict[str, Any]):
        self.necessity_analysis = verification_data.get("necessity_analysis", "")
        self.correctness_analysis = verification_data.get("correctness_analysis", "")
        self.efficiency_analysis = verification_data.get("efficiency_analysis", "")
        self.alternative_approaches = verification_data.get("alternative_approaches", "")
        self.critical_concerns = verification_data.get("critical_concerns", "")
        self.rating = verification_data.get("rating", 0)
        self.rating_justification = verification_data.get("rating_justification", "")
        self.regeneration_needed = verification_data.get("regeneration_needed", False)
        self.suggested_improvement = verification_data.get("suggested_improvement", "")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "necessity_analysis": self.necessity_analysis,
            "correctness_analysis": self.correctness_analysis,
            "efficiency_analysis": self.efficiency_analysis,
            "alternative_approaches": self.alternative_approaches,
            "critical_concerns": self.critical_concerns,
            "rating": self.rating,
            "rating_justification": self.rating_justification,
            "regeneration_needed": self.regeneration_needed,
            "suggested_improvement": self.suggested_improvement
        }


class TraceVerifier:
    """Verifier for evaluating reasoning trace steps."""

    def __init__(self,
                 llm_client: Optional[OpenAIClient] = None,
                 min_acceptable_rating: float = 6.0):
        """
        Initialize the trace verifier.

        Args:
            llm_client: LLM client for verification
            min_acceptable_rating: Minimum rating threshold for acceptable steps
        """
        self.llm_client = llm_client or OpenAIClient()
        self.min_acceptable_rating = min_acceptable_rating
        self.verification_history = []

        logger.info(f"TraceVerifier initialized with min rating threshold: {min_acceptable_rating}")

    def verify_step(self,
                   step_content: str,
                   step_index: int,
                   full_trace: List[Dict[str, Any]],
                   question: str,
                   image_path: Optional[Path] = None) -> VerificationResult:
        """
        Verify a single reasoning step.

        Args:
            step_content: Content of the step to verify
            step_index: Index of the step in the trace
            full_trace: Complete trace up to this point
            question: Original question
            image_path: Path to the image (optional)

        Returns:
            VerificationResult containing the verification assessment
        """
        logger.info(f"Verifying step {step_index}: {step_content[:100]}...")

        try:
            # Create verification prompt
            verification_messages = self._create_verification_messages(
                step_content, step_index, full_trace, question, image_path
            )

            # Get verification response
            response = self.llm_client.create_chat_completion(
                messages=verification_messages,
                max_tokens=500,
                temperature=0.1,  # Low temperature for consistent evaluation
                response_format={"type": "json_object"}
            )

            if not response:
                logger.error(f"Failed to get verification response for step {step_index}")
                return self._create_error_result("Failed to get LLM response")

            # Parse verification result
            verification_data = json.loads(response)
            result = VerificationResult(verification_data)

            # Store in history
            self.verification_history.append({
                "step_index": step_index,
                "timestamp": datetime.now().isoformat(),
                "result": result.to_dict()
            })

            logger.info(f"Step {step_index} verified - Rating: {result.rating}/10, "
                       f"Regeneration needed: {result.regeneration_needed}")

            return result

        except Exception as e:
            logger.error(f"Error verifying step {step_index}: {e}")
            return self._create_error_result(f"Verification error: {e}")

    def _create_verification_messages(self,
                                    step_content: str,
                                    step_index: int,
                                    full_trace: List[Dict[str, Any]],
                                    question: str,
                                    image_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Create messages for step verification."""

        # Get verifier system prompt
        system_prompt = prompt_manager.get_system_prompt("verifier")

        # Build context information
        context_info = {
            "original_question": question,
            "step_index": step_index,
            "total_steps_so_far": len([msg for msg in full_trace if msg.get("role") == "assistant"]),
            "current_step": step_content,
            "previous_context": self._extract_previous_context(full_trace, step_index)
        }

        # Create user message with context
        user_content = [
            {
                "type": "text",
                "text": f"""Please verify this reasoning step:

**Original Question:** {question}

**Step {step_index}:** {step_content}

**Previous Context:**
{context_info['previous_context']}

**Total Steps So Far:** {context_info['total_steps_so_far']}

Provide your verification assessment in the required JSON format."""
            }
        ]

        # Add image if available
        if image_path and image_path.exists():
            base64_image = encode_image_to_base64(image_path)
            if base64_image:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image,
                        "detail": "low"  # Use low detail to save tokens
                    }
                })

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

    def _extract_previous_context(self, full_trace: List[Dict[str, Any]], current_step_index: int) -> str:
        """Extract relevant previous context from the trace."""
        context_parts = []
        assistant_step_count = 0

        for msg in full_trace:
            if msg.get("role") == "assistant":
                assistant_step_count += 1
                if assistant_step_count >= current_step_index:
                    break

                content = msg.get("content", "")
                try:
                    parsed = json.loads(content)

                    # Handle new format (reasoning + action)
                    if "reasoning" in parsed and "action" in parsed:
                        action = parsed.get("action", "unknown")
                        reasoning = parsed.get("reasoning", "")

                        if action == "tool_call":
                            tool = parsed.get("tool_name", "unknown")
                            context_parts.append(f"Step {assistant_step_count}: [Reasoning] {reasoning}")
                            context_parts.append(f"Step {assistant_step_count}: [Tool Call] {tool}")
                        elif action == "final_answer":
                            answer = parsed.get("text", "")
                            context_parts.append(f"Step {assistant_step_count}: [Reasoning] {reasoning}")
                            context_parts.append(f"Step {assistant_step_count}: [Final Answer] {answer}")

                    # Handle old format for backward compatibility
                    else:
                        action = parsed.get("action", "unknown")
                        if action == "reasoning":
                            text = parsed.get("text", "")
                            context_parts.append(f"Step {assistant_step_count}: [Reasoning] {text}")
                        elif action == "tool_call":
                            tool = parsed.get("tool_name", "unknown")
                            context_parts.append(f"Step {assistant_step_count}: [Tool Call] {tool}")
                        elif action == "final_answer":
                            answer = parsed.get("text", "")
                            context_parts.append(f"Step {assistant_step_count}: [Final Answer] {answer}")

                except (json.JSONDecodeError, KeyError):
                    context_parts.append(f"Step {assistant_step_count}: [Unparseable] {content[:100]}")
            elif msg.get("role") == "user" and "Tool output:" in msg.get("content", ""):
                # Add tool results context
                tool_output = msg.get("content", "").replace("Tool output: ", "")
                context_parts.append(f"Tool Result: {tool_output[:100]}")

        return "\n".join(context_parts[-5:])  # Keep last 5 context items

    def _create_error_result(self, error_message: str) -> VerificationResult:
        """Create an error verification result."""
        return VerificationResult({
            "necessity_analysis": f"Error during verification: {error_message}",
            "correctness_analysis": "Could not assess due to verification error",
            "efficiency_analysis": "Could not assess due to verification error",
            "alternative_approaches": "Could not assess due to verification error",
            "critical_concerns": f"Verification failed: {error_message}",
            "rating": 1,  # Lowest rating for errors
            "rating_justification": f"Failed verification: {error_message}",
            "regeneration_needed": True,
            "suggested_improvement": "Fix verification system and retry"
        })

    def should_regenerate_step(self, verification_result: VerificationResult) -> bool:
        """
        Determine if a step should be regenerated based on verification.

        Args:
            verification_result: Result from step verification

        Returns:
            True if step should be regenerated
        """
        # Check explicit regeneration flag
        if verification_result.regeneration_needed:
            return True

        # Check rating threshold
        if verification_result.rating < self.min_acceptable_rating:
            logger.info(f"Step rating {verification_result.rating} below threshold {self.min_acceptable_rating}")
            return True

        return False

    def get_verification_history(self) -> List[Dict[str, Any]]:
        """Get the complete verification history."""
        return self.verification_history.copy()

    def save_verification_history(self, output_path: Path) -> bool:
        """
        Save verification history to file.

        Args:
            output_path: Path to save the verification history

        Returns:
            True if successful
        """
        try:
            verification_data = {
                "timestamp": datetime.now().isoformat(),
                "min_acceptable_rating": self.min_acceptable_rating,
                "verification_history": self.verification_history
            }

            with open(output_path, 'w') as f:
                json.dump(verification_data, f, indent=2)

            logger.info(f"Verification history saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving verification history: {e}")
            return False

    def clear_history(self):
        """Clear the verification history."""
        self.verification_history.clear()
        logger.info("Verification history cleared")
