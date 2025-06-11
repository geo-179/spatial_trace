
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..utils import save_reasoning_trace

logger = logging.getLogger(__name__)


class TraceProcessor:
    """Processor for analyzing and post-processing reasoning traces."""

    def __init__(self):
        """Initialize the trace processor."""
        pass

    def analyze_trace(self, trace: List[Dict[str, Any]], verification_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyze a reasoning trace and extract statistics, including verification data.

        Args:
            trace: Complete reasoning trace
            verification_data: Optional verification history

        Returns:
            Dictionary containing trace analysis
        """
        if not trace:
            return {"error": "Empty trace"}

        analysis = {
            "total_messages": len(trace),
            "step_count": 0,
            "tool_calls": [],
            "reasoning_steps": [],
            "final_answer": None,
            "success": False,
            "errors": []
        }

        for i, message in enumerate(trace):
            role = message.get("role", "unknown")
            content = message.get("content", "")

            if role == "assistant":
                try:
                    if isinstance(content, str) and content.strip():
                        parsed = json.loads(content)

                        if "reasoning" in parsed and "action" in parsed:
                            action = parsed.get("action")
                            reasoning = parsed.get("reasoning", "")

                            if action == "tool_call":
                                tool_name = parsed.get("tool_name")
                                analysis["tool_calls"].append({
                                    "step": analysis["step_count"],
                                    "tool": tool_name,
                                    "reasoning": reasoning,
                                    "message_index": i
                                })
                            elif action == "final_answer":
                                text = parsed.get("text", "")
                                analysis["final_answer"] = text
                                analysis["success"] = True

                            analysis["reasoning_steps"].append({
                                "step": analysis["step_count"],
                                "text": reasoning,
                                "action": action,
                                "message_index": i
                            })

                        else:
                            action = parsed.get("action")
                            if action == "tool_call":
                                tool_name = parsed.get("tool_name")
                                analysis["tool_calls"].append({
                                    "step": analysis["step_count"],
                                    "tool": tool_name,
                                    "message_index": i
                                })
                            elif action == "reasoning":
                                text = parsed.get("text", "")
                                analysis["reasoning_steps"].append({
                                    "step": analysis["step_count"],
                                    "text": text,
                                    "message_index": i
                                })
                            elif action == "final_answer":
                                text = parsed.get("text", "")
                                analysis["final_answer"] = text
                                analysis["success"] = True

                        analysis["step_count"] += 1

                except (json.JSONDecodeError, KeyError) as e:
                    analysis["errors"].append({
                        "message_index": i,
                        "error": f"Failed to parse assistant response: {e}"
                    })

        analysis["tool_call_count"] = len(analysis["tool_calls"])
        analysis["reasoning_step_count"] = len(analysis["reasoning_steps"])
        analysis["error_count"] = len(analysis["errors"])

        if analysis["tool_calls"]:
            tool_usage = {}
            for tool_call in analysis["tool_calls"]:
                tool = tool_call["tool"]
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
            analysis["tool_usage"] = tool_usage

        if verification_data:
            analysis["verification_summary"] = self._analyze_verification_data(verification_data)

        return analysis

    def _analyze_verification_data(self, verification_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze verification data and extract summary statistics."""
        if not verification_data:
            return {}

        ratings = [entry["result"]["rating"] for entry in verification_data]
        regenerations = [entry["result"]["regeneration_needed"] for entry in verification_data]

        return {
            "total_verifications": len(verification_data),
            "average_rating": sum(ratings) / len(ratings) if ratings else 0,
            "min_rating": min(ratings) if ratings else 0,
            "max_rating": max(ratings) if ratings else 0,
            "regenerations_needed": sum(regenerations),
            "regeneration_rate": sum(regenerations) / len(regenerations) if regenerations else 0,
            "ratings_distribution": {
                "excellent (9-10)": len([r for r in ratings if r >= 9]),
                "good (7-8)": len([r for r in ratings if 7 <= r < 9]),
                "acceptable (5-6)": len([r for r in ratings if 5 <= r < 7]),
                "poor (1-4)": len([r for r in ratings if r < 5])
            }
        }

    def extract_final_answer(self, trace: List[Dict[str, Any]]) -> Optional[str]:
        """
        Extract the final answer from a trace.

        Args:
            trace: Complete reasoning trace

        Returns:
            Final answer text or None if not found
        """
        for message in reversed(trace):
            if message.get("role") == "assistant":
                content = message.get("content", "")
                try:
                    if isinstance(content, str) and content.strip():
                        parsed = json.loads(content)

                        if parsed.get("action") == "final_answer":
                            return parsed.get("text")
                except (json.JSONDecodeError, KeyError):
                    continue
        return None

    def validate_trace_structure(self, trace: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate the structure of a reasoning trace.

        Args:
            trace: Reasoning trace to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        if not trace:
            issues.append("Trace is empty")
            return False, issues

        if trace[0].get("role") != "system":
            issues.append("First message should be system prompt")

        expected_role = "user"
        for i, message in enumerate(trace[1:], 1):
            role = message.get("role")
            if i == 1:
                if role != "user":
                    issues.append(f"Message {i} should be user message (initial question)")
            elif role not in ["user", "assistant"]:
                issues.append(f"Message {i} has invalid role: {role}")

        for i, message in enumerate(trace):
            if "role" not in message:
                issues.append(f"Message {i} missing 'role' field")
            if "content" not in message:
                issues.append(f"Message {i} missing 'content' field")

        for i, message in enumerate(trace):
            if message.get("role") == "assistant":
                content = message.get("content", "")
                if isinstance(content, str) and content.strip():
                    try:
                        parsed = json.loads(content)

                        if "reasoning" in parsed and "action" in parsed:
                            if not parsed.get("reasoning", "").strip():
                                issues.append(f"Assistant message {i} has empty reasoning field")
                            if parsed.get("action") not in ["tool_call", "final_answer"]:
                                issues.append(f"Assistant message {i} has invalid action")
                        elif "action" in parsed:
                            pass
                        else:
                            issues.append(f"Assistant message {i} missing required fields")

                    except json.JSONDecodeError:
                        issues.append(f"Assistant message {i} contains invalid JSON")

        return len(issues) == 0, issues

    def save_trace_with_analysis(
        self,
        trace: List[Dict[str, Any]],
        output_path: Path,
        question: str = None,
        image_path: Path = None,
        verification_data: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Save a trace with analysis metadata, including verification data.

        Args:
            trace: Complete reasoning trace
            output_path: Path to save the trace
            question: Original question (optional)
            image_path: Original image path (optional)
            verification_data: Verification history (optional)

        Returns:
            True if successful, False otherwise
        """
        try:
            analysis = self.analyze_trace(trace, verification_data)

            is_valid, issues = self.validate_trace_structure(trace)

            metadata = {
                "timestamp": datetime.now().isoformat(),
                "analysis": analysis,
                "validation": {
                    "is_valid": is_valid,
                    "issues": issues
                }
            }

            if question:
                metadata["question"] = question
            if image_path:
                metadata["image_path"] = str(image_path)
            if verification_data:
                metadata["verification_data"] = verification_data

            return save_reasoning_trace(trace, output_path, metadata)

        except Exception as e:
            logger.error(f"Error saving trace with analysis: {e}")
            return False

    def compare_traces(self, trace1: List[Dict[str, Any]], trace2: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare two reasoning traces.

        Args:
            trace1: First trace
            trace2: Second trace

        Returns:
            Comparison results
        """
        analysis1 = self.analyze_trace(trace1)
        analysis2 = self.analyze_trace(trace2)

        comparison = {
            "trace1_analysis": analysis1,
            "trace2_analysis": analysis2,
            "differences": {
                "length_diff": analysis1["total_messages"] - analysis2["total_messages"],
                "step_count_diff": analysis1["step_count"] - analysis2["step_count"],
                "tool_call_diff": analysis1["tool_call_count"] - analysis2["tool_call_count"],
                "both_successful": analysis1["success"] and analysis2["success"],
                "same_final_answer": analysis1.get("final_answer") == analysis2.get("final_answer")
            }
        }

        return comparison

    def create_trace_summary(self, trace: List[Dict[str, Any]]) -> str:
        """
        Create a human-readable summary of a trace.

        Args:
            trace: Complete reasoning trace

        Returns:
            Text summary of the trace
        """
        analysis = self.analyze_trace(trace)

        summary_parts = [
            f"Reasoning Trace Summary:",
            f"- Total messages: {analysis['total_messages']}",
            f"- Reasoning steps: {analysis['step_count']}",
            f"- Tool calls: {analysis['tool_call_count']}",
        ]

        if analysis.get("tool_usage"):
            summary_parts.append("- Tool usage:")
            for tool, count in analysis["tool_usage"].items():
                summary_parts.append(f"  - {tool}: {count} times")

        if analysis["success"]:
            summary_parts.append(f"- Final answer: {analysis['final_answer']}")
        else:
            summary_parts.append("- No final answer reached")

        if analysis["errors"]:
            summary_parts.append(f"- Errors encountered: {analysis['error_count']}")

        return "\n".join(summary_parts)
