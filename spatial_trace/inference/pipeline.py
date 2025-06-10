"""
Main spatial reasoning pipeline with integrated verification.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..llm_interface import OpenAIClient, OutputParser, ActionType, prompt_manager
from ..tools import tool_registry
from ..utils import encode_image_to_base64, validate_image_path
from .verifier import TraceVerifier, VerificationResult

logger = logging.getLogger(__name__)


class SpatialReasoningPipeline:
    """Main pipeline for spatial reasoning trace generation with verification."""

    def __init__(self,
                 llm_client: Optional[OpenAIClient] = None,
                 max_steps: int = 10,
                 enable_verification: bool = False,
                 min_acceptable_rating: float = 6.0,
                 max_regeneration_attempts: int = 3):
        """
        Initialize the spatial reasoning pipeline.

        Args:
            llm_client: LLM client to use (defaults to OpenAIClient)
            max_steps: Maximum number of reasoning steps
            enable_verification: Whether to enable step verification
            min_acceptable_rating: Minimum rating threshold for acceptable steps
            max_regeneration_attempts: Maximum number of regeneration attempts per step
        """
        self.llm_client = llm_client or OpenAIClient()
        self.max_steps = max_steps
        self.tool_registry = tool_registry
        self.prompt_manager = prompt_manager
        self.enable_verification = enable_verification
        self.max_regeneration_attempts = max_regeneration_attempts

        # Initialize verifier if enabled
        self.verifier = None
        if enable_verification:
            self.verifier = TraceVerifier(
                llm_client=self.llm_client,
                min_acceptable_rating=min_acceptable_rating
            )
            logger.info("Verification enabled in pipeline")

        # Configure tools from environment if not already done
        if not self.tool_registry.list_tools():
            logger.info("Configuring tools from environment variables")
            self.tool_registry.configure_from_env()

        logger.info(f"Pipeline initialized with {len(self.tool_registry.list_tools())} tools")

    def generate_reasoning_trace(
        self,
        question: str,
        image_path: Path,
        prompt_name: str = "spatial_reasoning"
    ) -> List[Dict[str, Any]]:
        """
        Generate a complete reasoning trace for a spatial question with optional verification.

        Args:
            question: The spatial reasoning question
            image_path: Path to the input image
            prompt_name: Name of the prompt template to use

        Returns:
            Complete reasoning trace as a list of message dictionaries
        """
        logger.info(f"Starting reasoning trace for question: '{question}'")
        logger.info(f"Image path: {image_path}")
        logger.info(f"Verification enabled: {self.enable_verification}")

        # Validate inputs
        if not self._validate_inputs(question, image_path):
            return []

        # Initialize the trace
        trace = self._initialize_trace(question, image_path, prompt_name)
        if not trace:
            logger.error("Failed to initialize reasoning trace")
            return []

        current_image_path = image_path

        # Main reasoning loop
        for step in range(self.max_steps):
            logger.info(f"Reasoning step {step + 1}/{self.max_steps}")

            # Generate step with potential regeneration
            success = self._generate_verified_step(trace, step, question, current_image_path)

            if not success:
                logger.error(f"Failed to generate valid step {step + 1}")
                break

            # Check if we have a final answer
            last_message = trace[-1] if trace else None
            if last_message and last_message.get("role") == "assistant":
                try:
                    parsed = json.loads(last_message.get("content", "{}"))
                    if parsed.get("action") == "final_answer":
                        logger.info("Pipeline completed with final answer")
                        break
                except (json.JSONDecodeError, KeyError):
                    pass

        else:
            logger.warning(f"Pipeline reached maximum steps ({self.max_steps}) without completion")

        logger.info(f"Reasoning trace completed with {len(trace)} messages")
        return trace

    def _generate_verified_step(self,
                              trace: List[Dict[str, Any]],
                              step_index: int,
                              question: str,
                              current_image_path: Path) -> bool:
        """
        Generate a single step with verification and potential regeneration.

        Args:
            trace: Current trace
            step_index: Current step index
            question: Original question
            current_image_path: Current image path

        Returns:
            True if step was successfully generated and verified
        """
        for attempt in range(self.max_regeneration_attempts + 1):
            if attempt > 0:
                logger.info(f"Regeneration attempt {attempt}/{self.max_regeneration_attempts} for step {step_index + 1}")

            # Get LLM response
            response_text = self._get_llm_response(trace)
            if not response_text:
                logger.error(f"Failed to get LLM response at step {step_index + 1}, attempt {attempt + 1}")
                continue

            # Parse the response
            success, parsed_data, error = OutputParser.parse_llm_response(response_text)
            if not success:
                logger.error(f"Failed to parse LLM response: {error}")
                if attempt < self.max_regeneration_attempts:
                    # Add error feedback and try again
                    trace.append({"role": "user", "content": f"Error: {error}. Please provide a valid JSON response."})
                    continue
                else:
                    return False

            # Verify step if verification is enabled
            if self.enable_verification:
                verification_result = self.verifier.verify_step(
                    step_content=response_text,
                    step_index=step_index + 1,
                    full_trace=trace,
                    question=question,
                    image_path=current_image_path
                )

                # Check if regeneration is needed
                if self.verifier.should_regenerate_step(verification_result):
                    logger.info(f"Step {step_index + 1} failed verification (rating: {verification_result.rating})")

                    if attempt < self.max_regeneration_attempts:
                        # Add improvement guidance and try again
                        improvement_message = self._create_improvement_message(verification_result)
                        trace.append(improvement_message)
                        continue
                    else:
                        logger.warning(f"Max regeneration attempts reached for step {step_index + 1}, accepting step")

            # Step is acceptable, add to trace and process
            trace.append({"role": "assistant", "content": response_text})

            # Process the action
            action_type, action_info = OutputParser.extract_action_info(parsed_data)

            if action_type == ActionType.FINAL_ANSWER:
                final_answer = action_info["text"]
                reasoning = action_info["reasoning"]
                logger.info(f"AI reasoning: {reasoning}")
                logger.info(f"Final answer generated: {final_answer}")
                return True

            elif action_type == ActionType.TOOL_CALL:
                tool_name = action_info["tool_name"]
                reasoning = action_info["reasoning"]
                logger.info(f"AI reasoning: {reasoning}")
                logger.info(f"AI requested tool: {tool_name}")

                # Execute tool
                tool_message, new_image_path = self._execute_tool(tool_name, current_image_path)

                # Add tool result to trace
                tool_result_message = self._create_tool_result_message(tool_message, new_image_path)
                trace.append(tool_result_message)
                return True

        return False

    def _create_improvement_message(self, verification_result: VerificationResult) -> Dict[str, Any]:
        """Create a feedback message for step improvement."""
        feedback_text = f"""Your previous step received a rating of {verification_result.rating}/10 and needs improvement.

**Issues identified:**
- {verification_result.critical_concerns}

**Suggested improvement:**
{verification_result.suggested_improvement}

Please provide a better response that addresses these concerns."""

        return {"role": "user", "content": feedback_text}

    def _validate_inputs(self, question: str, image_path: Path) -> bool:
        """Validate pipeline inputs."""
        if not question or not question.strip():
            logger.error("Question cannot be empty")
            return False

        if not validate_image_path(image_path):
            logger.error(f"Invalid image path: {image_path}")
            return False

        if not self.llm_client.is_available():
            logger.error("LLM client is not available")
            return False

        return True

    def _initialize_trace(
        self,
        question: str,
        image_path: Path,
        prompt_name: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Initialize the reasoning trace with system prompt and first user message."""
        try:
            # Encode the initial image
            base64_image = encode_image_to_base64(image_path)
            if not base64_image:
                logger.error("Failed to encode initial image")
                return None

            # Create initial messages
            messages = self.prompt_manager.create_initial_messages(
                question, base64_image, prompt_name
            )

            logger.debug(f"Initialized trace with {len(messages)} messages")
            return messages

        except Exception as e:
            logger.error(f"Error initializing trace: {e}")
            return None

    def _get_llm_response(self, trace: List[Dict[str, Any]]) -> Optional[str]:
        """Get response from LLM."""
        try:
            response = self.llm_client.create_chat_completion(
                messages=trace,
                max_tokens=200,
                temperature=0.5,
                response_format={"type": "json_object"}
            )
            return response
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            return None

    def _execute_tool(self, tool_name: str, image_path: Path) -> tuple[str, Optional[Path]]:
        """Execute a tool and return the result."""
        try:
            return self.tool_registry.run_tool(tool_name, image_path)
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {e}"
            logger.error(error_msg)
            return error_msg, None

    def _create_tool_result_message(
        self,
        tool_message: str,
        new_image_path: Optional[Path]
    ) -> Dict[str, Any]:
        """Create a message for tool results."""
        if new_image_path:
            # Encode the new image
            new_base64_image = encode_image_to_base64(new_image_path)
            if new_base64_image:
                return self.prompt_manager.create_tool_result_message(
                    tool_message, new_base64_image
                )

        # Fallback to text-only message
        return self.prompt_manager.create_tool_result_message(tool_message)

    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        return self.tool_registry.list_tools()

    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of available tools."""
        return self.tool_registry.get_tool_descriptions()

    def check_system_status(self) -> Dict[str, Any]:
        """Check the status of the pipeline system."""
        status = {
            "llm_available": self.llm_client.is_available(),
            "llm_info": self.llm_client.get_model_info(),
            "available_tools": self.tool_registry.list_tools(),
            "tool_availability": self.tool_registry.check_tool_availability(),
            "max_steps": self.max_steps,
            "verification_enabled": self.enable_verification
        }

        if self.verifier:
            status["verifier_min_rating"] = self.verifier.min_acceptable_rating
            status["max_regeneration_attempts"] = self.max_regeneration_attempts

        return status

    def get_verification_history(self) -> Optional[List[Dict[str, Any]]]:
        """Get verification history if verifier is enabled."""
        if self.verifier:
            return self.verifier.get_verification_history()
        return None

    def save_verification_history(self, output_path: Path) -> bool:
        """Save verification history if verifier is enabled."""
        if self.verifier:
            return self.verifier.save_verification_history(output_path)
        return False
