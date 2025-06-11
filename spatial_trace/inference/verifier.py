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
                   image_path: Optional[Path] = None,
                   attempt_number: int = 1) -> VerificationResult:
        """Enhanced verification with retry logic and robust error handling."""

        logger.info(f"Verifying step {step_index}, attempt {attempt_number}: {step_content[:100]}...")

        # Retry logic for API failures
        max_retries = 3
        for retry in range(max_retries + 1):
            try:
                # Create verification prompt
                verification_messages = self._create_verification_messages(
                    step_content, step_index, full_trace, question, image_path
                )

                # Add context length check
                total_chars = sum(len(str(msg)) for msg in verification_messages)
                if total_chars > 50000:  # Rough token limit check
                    logger.warning(f"Verification context very long ({total_chars} chars), may cause API issues")

                # Get verification response with timeout consideration
                response = self.llm_client.create_chat_completion(
                    messages=verification_messages,
                    max_tokens=500,
                    temperature=0.1,  # Low temperature for consistent evaluation
                    response_format={"type": "json_object"}
                )

                if not response or len(response.strip()) == 0:
                    if retry < max_retries:
                        logger.warning(f"Empty verification response for step {step_index}, retry {retry + 1}/{max_retries}")
                        continue
                    else:
                        logger.error(f"Failed to get verification response for step {step_index} after {max_retries + 1} attempts")
                        return self._create_fallback_result(step_content, "API failure after retries")

                # Parse verification result
                try:
                    verification_data = json.loads(response)
                except json.JSONDecodeError as e:
                    if retry < max_retries:
                        logger.warning(f"JSON parse error for step {step_index}, retry {retry + 1}/{max_retries}: {e}")
                        continue
                    else:
                        logger.error(f"JSON parse failed for step {step_index} after retries: {e}")
                        logger.error(f"Raw response: {response[:200]}...")
                        return self._create_fallback_result(step_content, f"JSON parse error: {e}")

                # Validate required fields
                if not self._validate_verification_data(verification_data):
                    if retry < max_retries:
                        logger.warning(f"Invalid verification data for step {step_index}, retry {retry + 1}")
                        continue
                    else:
                        logger.error(f"Invalid verification data structure for step {step_index}")
                        return self._create_fallback_result(step_content, "Invalid verification structure")

                result = VerificationResult(verification_data)

                # Determine the authoritative regeneration decision based on our threshold.
                authoritative_regeneration_needed = self.should_regenerate_step(result)

                # Log both the LLM's suggestion and our final decision for clarity.
                logger.info(
                    f"Step {step_index} verified - Rating: {result.rating}/10, "
                    f"LLM Suggestion: {result.regeneration_needed}, "
                    f"ACTUAL REGENERATION: {authoritative_regeneration_needed}"
                )

                # CRITICAL: Ensure the returned result object has the correct, authoritative value.
                result.regeneration_needed = authoritative_regeneration_needed

                # Enhanced history entry with attempt tracking
                self.verification_history.append({
                    "step_index": step_index,
                    "attempt_number": attempt_number,
                    "timestamp": datetime.now().isoformat(),
                    "step_content": step_content[:200],  # Save snippet for reference
                    "result": result.to_dict(),
                    "passed_threshold": result.rating >= self.min_acceptable_rating,
                    "regeneration_triggered": authoritative_regeneration_needed,
                    "verification_retries": retry
                })

                return result

            except Exception as e:
                if retry < max_retries:
                    logger.warning(f"Verification error for step {step_index}, retry {retry + 1}/{max_retries}: {e}")
                    continue
                else:
                    logger.error(f"Error verifying step {step_index} after retries: {e}")
                    return self._create_fallback_result(step_content, f"Verification error: {e}")

        # This should never be reached due to the return statements above
        return self._create_fallback_result(step_content, "Unexpected verification flow")

    def _validate_verification_data(self, data: Dict[str, Any]) -> bool:
        """Validate that verification response has required fields."""
        required_fields = ["rating", "rating_justification", "necessity_analysis", "correctness_analysis"]
        return all(field in data for field in required_fields) and isinstance(data.get("rating"), (int, float))

    def _create_fallback_result(self, step_content: str, error_message: str) -> VerificationResult:
        """Create a reasonable fallback result when verification fails (NOT rating 1)."""

        # Simple heuristic evaluation when verification API fails
        fallback_rating = self._heuristic_evaluation(step_content)

        return VerificationResult({
            "necessity_analysis": f"Verification failed: {error_message}. Using fallback evaluation.",
            "correctness_analysis": "Could not verify due to API issues. Using heuristic assessment.",
            "efficiency_analysis": "Could not assess efficiency due to verification failure.",
            "alternative_approaches": "Could not assess alternatives due to verification failure.",
            "critical_concerns": f"Verification system failed: {error_message}",
            "rating": fallback_rating,
            "rating_justification": f"Fallback rating due to verification failure. Heuristic assessment: {fallback_rating}/10",
            "regeneration_needed": fallback_rating < self.min_acceptable_rating,
            "suggested_improvement": "Verification system needs fixing - using fallback evaluation"
        })

    def _heuristic_evaluation(self, step_content: str) -> float:
        """Simple heuristic evaluation when verification API fails."""
        try:
            # Parse step content to do basic validation
            parsed = json.loads(step_content)

            # Check for required fields
            if "reasoning" not in parsed or "action" not in parsed:
                return 4.0  # Below threshold but not catastrophic

            reasoning = parsed.get("reasoning", "")
            action = parsed.get("action", "")

            # Basic quality checks
            rating = 6.0  # Start with acceptable baseline

            # Reasoning quality heuristics
            if len(reasoning) < 20:
                rating -= 1.0  # Too brief
            elif len(reasoning) > 500:
                rating += 0.5  # Detailed reasoning

            # Action validation
            if action in ["tool_call", "final_answer"]:
                rating += 0.5  # Valid action

            if action == "tool_call":
                tool_name = parsed.get("tool_name", "")
                if tool_name in ["sam2", "dav2", "trellis"]:
                    rating += 0.5  # Valid tool

            return min(max(rating, 1.0), 10.0)  # Clamp to 1-10 range

        except (json.JSONDecodeError, KeyError):
            return 5.0  # Neutral rating for unparseable content

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

    def should_regenerate_step(self, verification_result: VerificationResult) -> bool:
        """
        Determine if a step should be regenerated based ONLY on the rating threshold.
        The LLM's suggestion is ignored to enforce our quality gate.

        Args:
            verification_result: Result from step verification

        Returns:
            True if step should be regenerated
        """
        # The decision is made entirely based on the numeric rating.
        if verification_result.rating < self.min_acceptable_rating:
            # The log message was moved to verify_step for better context.
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
