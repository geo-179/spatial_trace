import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import shutil

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
                 enable_verification: bool = True,
                 enable_regeneration: bool = True,
                 min_acceptable_rating: float = 8.0,
                 max_regeneration_attempts: int = 3):
        """
        Initialize the spatial reasoning pipeline.

        Args:
            llm_client: LLM client to use (defaults to OpenAIClient)
            max_steps: Maximum number of reasoning steps
            enable_verification: Whether to enable step verification
            enable_regeneration: Whether to enable step regeneration
            min_acceptable_rating: Minimum rating threshold for acceptable steps
            max_regeneration_attempts: Maximum number of regeneration attempts per step
        """
        self.llm_client = llm_client or OpenAIClient()
        self.max_steps = max_steps
        self.tool_registry = tool_registry
        self.prompt_manager = prompt_manager
        self.enable_verification = enable_verification
        self.enable_regeneration = enable_regeneration
        self.max_regeneration_attempts = max_regeneration_attempts
        self.current_trace_tool_images = []  # Track tool images for current trace

        self.verifier = TraceVerifier(min_acceptable_rating=min_acceptable_rating)
        if enable_verification:
            logger.info("Verification enabled in pipeline")

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
        self.current_trace_tool_images = []

        logger.info(f"Starting reasoning trace for question: '{question}'")
        logger.info(f"Image path: {image_path}")
        logger.info(f"Verification enabled: {self.enable_verification}")

        if not self._validate_inputs(question, image_path):
            return []

        trace = self._initialize_trace(question, image_path, prompt_name)
        if not trace:
            logger.error("Failed to initialize reasoning trace")
            return []

        current_image_path = image_path

        for step in range(self.max_steps):
            logger.info(f"Reasoning step {step + 1}/{self.max_steps}")

            success = self._generate_verified_step(trace, step, question, current_image_path)

            if not success:
                logger.error(f"Failed to generate valid step {step + 1}")
                break

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
        logger.info(f"Tool images generated: {len(self.current_trace_tool_images)}")

        self._track_tool_call_statistics(trace)

        return trace

    def _generate_verified_step(self,
                              trace: List[Dict[str, Any]],
                              step_index: int,
                              question: str,
                              current_image_path: Path) -> bool:
        """Enhanced to track all verification attempts."""

        last_verification_result = None

        for attempt in range(self.max_regeneration_attempts + 1):
            if attempt > 0:
                logger.info(f"Regeneration attempt {attempt}/{self.max_regeneration_attempts} for step {step_index + 1}")

            working_trace = trace.copy()

            if attempt > 0 and self.enable_verification and last_verification_result:
                improvement_message = self._create_improvement_message(last_verification_result)
                working_trace.append(improvement_message)

            if not response_text:
                logger.error(f"Failed to get LLM response at step {step_index + 1}, attempt {attempt + 1}")
                continue

            success, parsed_data, error = OutputParser.parse_llm_response(response_text)
            if not success:
                logger.error(f"Failed to parse LLM response: {error}")
                if attempt < self.max_regeneration_attempts:
                    error_message = {"role": "user", "content": f"Error: {error}. Please provide a valid JSON response."}
                    working_trace.append(error_message)
                    continue
                else:
                    return False

            if self.enable_verification:
                verification_result = self.verifier.verify_step(
                    step_content=response_text,
                    step_index=step_index + 1,
                    full_trace=trace,
                    question=question,
                    image_path=current_image_path,
                    attempt_number=attempt + 1  
                )

                last_verification_result = verification_result

                if self.verifier.should_regenerate_step(verification_result):
                    logger.info(f"Step {step_index + 1} failed verification (rating: {verification_result.rating})")

                    if attempt < self.max_regeneration_attempts:
                        continue
                    else:
                        logger.warning(f"Max attempts reached, accepting step with rating {verification_result.rating}")

            trace.append({"role": "assistant", "content": response_text})

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

                tool_message, tool_image_path = self._execute_tool(tool_name, current_image_path)

                if tool_image_path and tool_image_path.exists():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    unique_filename = f"{tool_name.lower()}_{step_index + 1}_{attempt + 1}_{timestamp}.png"
                    safe_tool_path = tool_image_path.parent / unique_filename

                    try:
                        shutil.copy2(tool_image_path, safe_tool_path)
                        logger.info(f"Safely copied tool output: {safe_tool_path}")

                        tool_image_info = {
                            "step_index": step_index + 1,
                            "attempt": attempt + 1,
                            "tool_name": tool_name,
                            "source_path": str(safe_tool_path),
                            "reasoning": reasoning,
                            "timestamp": datetime.now().isoformat()
                        }
                        self.current_trace_tool_images.append(tool_image_info)

                    except Exception as e:
                        logger.error(f"Failed to copy tool image: {e}")
                        tool_image_info = {
                            "step_index": step_index + 1,
                            "attempt": attempt + 1,
                            "tool_name": tool_name,
                            "source_path": str(tool_image_path),
                            "reasoning": reasoning,
                            "timestamp": datetime.now().isoformat()
                        }
                        self.current_trace_tool_images.append(tool_image_info)

                tool_result_message = self._create_tool_result_message(tool_message, tool_image_path)
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
            base64_image = encode_image_to_base64(image_path)
            if not base64_image:
                logger.error("Failed to encode initial image")
                return None

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

        multimodal_count = sum(1 for msg in trace if isinstance(msg.get("content"), list))
        logger.info(f"Sending {len(trace)} messages ({multimodal_count} with images) to LLM")

        try:
            response = self.llm_client.create_chat_completion(
                messages=trace,
                max_tokens=200,
                temperature=0.5,
                response_format={"type": "json_object"}
            )
            return response
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
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

        if new_image_path and new_image_path.exists():
            new_base64_image = encode_image_to_base64(new_image_path)

            if new_base64_image:
                message = self.prompt_manager.create_tool_result_message(tool_message, new_base64_image)
                logger.info("Tool output with image included")
                return message
            else:
                logger.error(f"Failed to encode tool image: {new_image_path}")

        logger.info("Tool output text-only")
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
            "verification_enabled": self.enable_verification,
            "regeneration_enabled": self.enable_regeneration
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

    def get_current_trace_tool_images(self) -> List[Dict[str, Any]]:
        """Get tool images generated in current trace."""
        return self.current_trace_tool_images.copy()

    def _track_tool_call_statistics(self, trace: List[Dict[str, Any]]):
        """Log statistics about tool calls and image inclusion."""
        total_tools = len(self.current_trace_tool_images)
        if total_tools > 0:
            by_tool = {}
            for tool_info in self.current_trace_tool_images:
                tool_name = tool_info["tool_name"]
                by_tool[tool_name] = by_tool.get(tool_name, 0) + 1

            tool_summary = ", ".join(f"{tool}: {count}" for tool, count in by_tool.items())
            logger.info(f"Tool calls: {tool_summary}")

        text_only_messages = 0
        multimodal_messages = 0
        for message in trace:
            content = message.get("content")
            if isinstance(content, list):
                multimodal_messages += 1
            elif isinstance(content, str):
                text_only_messages += 1

        image_inclusion_rate = multimodal_messages/len(trace) * 100 if len(trace) > 0 else 0
        logger.info(f"Final trace: {len(trace)} messages, {image_inclusion_rate:.1f}% with images")

    def verify_and_regenerate_step_if_needed(self, trace: List[Dict[str, Any]], last_response: str, step_index: int, question: str, image_path: Path) -> Tuple[str, bool]:
        """
        Verify a step. If it's good enough, accept it. If not, run a regeneration loop
        and select the best response from all attempts within that loop.
        """
        if not self.enable_verification:
            return last_response, True

        initial_verification_result = self.verifier.verify_step(
            step_content=last_response,
            step_index=step_index,
            full_trace=trace,
            question=question,
            image_path=image_path,
            attempt_number=1
        )

        if not self.enable_regeneration or not self.verifier.should_regenerate_step(initial_verification_result):
            return last_response, True

        logger.info(f"Step {step_index} failed verification (rating: {initial_verification_result.rating}). Starting regeneration loop to find best response.")

        step_attempts = [{
            "response": last_response,
            "rating": initial_verification_result.rating
        }]

        working_trace = trace[:-1]

        for attempt in range(self.max_regeneration_attempts):
            logger.info(f"Regeneration attempt {attempt + 1}/{self.max_regeneration_attempts} for step {step_index}")

            last_attempt_result = self.verifier.get_verification_history()[-1]
            error_message = self.prompt_manager.create_verification_feedback_message(last_attempt_result['result'])

            temp_trace = working_trace + [error_message]
            new_response = self._get_llm_response(temp_trace)

            if not new_response:
                logger.error("Failed to get a new response from LLM during regeneration.")
                continue

            new_verification_result = self.verifier.verify_step(
                step_content=new_response,
                step_index=step_index,
                full_trace=working_trace,
                question=question,
                image_path=image_path,
                attempt_number=attempt + 2
            )

            step_attempts.append({
                "response": new_response,
                "rating": new_verification_result.rating
            })

            if new_verification_result.rating == 10:
                logger.info("Achieved perfect score (10/10) on regeneration. Selecting this response.")
                break

        best_attempt = max(step_attempts, key=lambda x: x["rating"])

        logger.info(f"Regeneration complete. Selected best response for step {step_index} with rating {best_attempt['rating']}.")

        return best_attempt["response"], True
