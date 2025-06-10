"""
Grade reasoning traces on multiple dimensions.
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TraceGrade:
    """Simple grade structure."""
    trace_id: str
    overall_score: float  # 0-10
    reasoning_quality: float  # How logical are the reasoning steps?
    tool_usage: float  # Appropriate tool selection?
    correctness: float  # Is the final answer correct?
    efficiency: float  # How efficiently was the problem solved?
    feedback: str


class TraceGrader:
    """Simple but comprehensive trace grading system."""

    def __init__(self):
        self.weights = {
            "reasoning_quality": 0.3,
            "tool_usage": 0.2,
            "correctness": 0.4,  # Most important
            "efficiency": 0.1
        }

    def grade_trace(self,
                   trace_data: Dict[str, Any],
                   ground_truth: str = None) -> TraceGrade:
        """
        Grade a single trace comprehensively.

        Args:
            trace_data: Full trace data including trace and metadata
            ground_truth: Correct answer (optional)

        Returns:
            Grade with scores and feedback
        """
        trace_id = trace_data.get("id", "unknown")
        trace = trace_data.get("trace", [])

        # Extract key information
        reasoning_steps = self._extract_reasoning_steps(trace)
        tool_calls = self._extract_tool_calls(trace)
        final_answer = self._extract_final_answer(trace)

        # Grade each dimension
        reasoning_score = self._grade_reasoning_quality(reasoning_steps)
        tool_score = self._grade_tool_usage(tool_calls, reasoning_steps)
        correctness_score = self._grade_correctness(final_answer, ground_truth)
        efficiency_score = self._grade_efficiency(len(reasoning_steps), len(tool_calls))

        # Calculate weighted overall score
        overall_score = (
            reasoning_score * self.weights["reasoning_quality"] +
            tool_score * self.weights["tool_usage"] +
            correctness_score * self.weights["correctness"] +
            efficiency_score * self.weights["efficiency"]
        )

        # Generate feedback
        feedback = self._generate_feedback(
            reasoning_score, tool_score, correctness_score, efficiency_score
        )

        return TraceGrade(
            trace_id=trace_id,
            overall_score=overall_score,
            reasoning_quality=reasoning_score,
            tool_usage=tool_score,
            correctness=correctness_score,
            efficiency=efficiency_score,
            feedback=feedback
        )

    def _extract_reasoning_steps(self, trace: List[Dict]) -> List[str]:
        """Extract reasoning text from assistant messages."""
        reasoning_steps = []
        for message in trace:
            if message.get("role") == "assistant":
                try:
                    content = json.loads(message.get("content", "{}"))
                    if "reasoning" in content:
                        reasoning_steps.append(content["reasoning"])
                except json.JSONDecodeError:
                    continue
        return reasoning_steps

    def _extract_tool_calls(self, trace: List[Dict]) -> List[str]:
        """Extract tool calls from trace."""
        tools = []
        for message in trace:
            if message.get("role") == "assistant":
                try:
                    content = json.loads(message.get("content", "{}"))
                    if content.get("action") == "tool_call":
                        tools.append(content.get("tool_name", "unknown"))
                except json.JSONDecodeError:
                    continue
        return tools

    def _extract_final_answer(self, trace: List[Dict]) -> Optional[str]:
        """Extract final answer from trace."""
        for message in reversed(trace):
            if message.get("role") == "assistant":
                try:
                    content = json.loads(message.get("content", "{}"))
                    if content.get("action") == "final_answer":
                        return content.get("text", "").strip()
                except json.JSONDecodeError:
                    continue
        return None

    def _grade_reasoning_quality(self, reasoning_steps: List[str]) -> float:
        """Grade reasoning quality (0-10)."""
        if not reasoning_steps:
            return 0.0

        score = 5.0  # Base score

        # Check for logical progression
        if len(reasoning_steps) > 1:
            score += 1.0

        # Check for detail and clarity (rough heuristic)
        avg_length = sum(len(step) for step in reasoning_steps) / len(reasoning_steps)
        if avg_length > 50:  # Reasonably detailed
            score += 2.0
        elif avg_length > 20:
            score += 1.0

        # Check for specific spatial reasoning keywords
        spatial_keywords = ["shape", "color", "position", "size", "compare", "identify"]
        for step in reasoning_steps:
            if any(keyword in step.lower() for keyword in spatial_keywords):
                score += 0.5
                break

        return min(score, 10.0)

    def _grade_tool_usage(self, tool_calls: List[str], reasoning_steps: List[str]) -> float:
        """Grade tool usage appropriateness (0-10)."""
        if not tool_calls:
            return 5.0 if len(reasoning_steps) == 1 else 3.0  # May not need tools

        score = 6.0  # Base for using tools

        # Appropriate tool selection (sam2 for segmentation is good)
        if "sam2" in tool_calls:
            score += 2.0

        # Not too many tools (efficiency)
        if len(tool_calls) <= 2:
            score += 1.0

        # Tool use matches reasoning
        if tool_calls and any("segment" in step.lower() or "identify" in step.lower()
                             for step in reasoning_steps):
            score += 1.0

        return min(score, 10.0)

    def _grade_correctness(self, final_answer: str, ground_truth: str) -> float:
        """Grade answer correctness (0-10)."""
        if not final_answer:
            return 0.0

        if not ground_truth:
            return 5.0  # Neutral if no ground truth

        # Simple exact match (case-insensitive)
        if final_answer.lower().strip() == ground_truth.lower().strip():
            return 10.0

        # Partial credit for yes/no questions
        if ground_truth.lower() in ["yes", "no"]:
            if final_answer.lower().startswith(ground_truth.lower()):
                return 8.0

        return 0.0

    def _grade_efficiency(self, num_reasoning_steps: int, num_tool_calls: int) -> float:
        """Grade efficiency (0-10)."""
        total_steps = num_reasoning_steps + num_tool_calls

        if total_steps <= 2:
            return 10.0  # Very efficient
        elif total_steps <= 4:
            return 8.0   # Good
        elif total_steps <= 6:
            return 6.0   # Acceptable
        else:
            return 4.0   # Too many steps

    def _generate_feedback(self, reasoning: float, tools: float,
                          correctness: float, efficiency: float) -> str:
        """Generate human-readable feedback."""
        feedback_parts = []

        if reasoning >= 8:
            feedback_parts.append("Strong reasoning quality")
        elif reasoning <= 4:
            feedback_parts.append("Reasoning needs improvement")

        if tools >= 8:
            feedback_parts.append("Good tool usage")
        elif tools <= 4:
            feedback_parts.append("Poor tool selection")

        if correctness >= 8:
            feedback_parts.append("Correct answer")
        elif correctness == 0:
            feedback_parts.append("Incorrect answer")

        if efficiency >= 8:
            feedback_parts.append("Efficient solution")
        elif efficiency <= 4:
            feedback_parts.append("Too many steps")

        return "; ".join(feedback_parts) if feedback_parts else "Average performance"

    def grade_dataset(self, traces_path: Path, ground_truth_path: Path = None) -> List[TraceGrade]:
        """Grade multiple traces."""
        # Load traces
        with open(traces_path, 'r') as f:
            traces = json.load(f)

        # Load ground truth if available
        ground_truth = {}
        if ground_truth_path and ground_truth_path.exists():
            with open(ground_truth_path, 'r') as f:
                ground_truth = json.load(f)

        grades = []
        for trace_data in traces:
            trace_id = trace_data.get("id", "unknown")
            gt = ground_truth.get(trace_id)
            grade = self.grade_trace(trace_data, gt)
            grades.append(grade)

        return grades

    def calculate_summary_stats(self, grades: List[TraceGrade]) -> Dict[str, float]:
        """Calculate summary statistics."""
        if not grades:
            return {}

        return {
            "average_overall": sum(g.overall_score for g in grades) / len(grades),
            "average_reasoning": sum(g.reasoning_quality for g in grades) / len(grades),
            "average_tool_usage": sum(g.tool_usage for g in grades) / len(grades),
            "average_correctness": sum(g.correctness for g in grades) / len(grades),
            "average_efficiency": sum(g.efficiency for g in grades) / len(grades),
            "total_graded": len(grades)
        }


if __name__ == "__main__":
    grader = TraceGrader()

    # Example: grade quality traces
    if Path("evaluation/quality_traces.json").exists():
        grades = grader.grade_dataset(Path("evaluation/quality_traces.json"))
        stats = grader.calculate_summary_stats(grades)

        print("Grading Results:")
        for key, value in stats.items():
            print(f"{key}: {value:.2f}")

        # Show detailed grades for first few traces
        for grade in grades[:3]:
            print(f"\nTrace {grade.trace_id}:")
            print(f"  Overall: {grade.overall_score:.1f}/10")
            print(f"  Feedback: {grade.feedback}")
