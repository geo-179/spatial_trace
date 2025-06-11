import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
import argparse
import sys
import copy
import time

sys.path.append(str(Path(__file__).parent.parent))
from spatial_trace.llm_interface.openai_client import OpenAIClient

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
    """
    Uses an LLM to grade reasoning traces based on quality and diversity criteria.
    """

    def __init__(self):
        """Initializes the grader with an LLM client and a system prompt."""
        self.llm_client = OpenAIClient()
        self.system_prompt = self._get_system_prompt()
        self.image_keys = {'image_base64', 'image', 'mask', 'masked_image', 'depth_map'}

    def _get_system_prompt(self) -> str:
        """Defines the critical instructions for the LLM judge."""
        return """
You are an expert evaluator of AI-generated reasoning traces. Your task is to determine if a given reasoning trace is a "beneficial" example for a dataset focused on complex spatial reasoning processes.

Your evaluation should focus ONLY on two criteria:
1.  **Solid Reasoning:** Is the AI's reasoning clear, logical, and not repetitive? Does it show a chain of thought that builds upon previous steps and tool outputs?
2.  **Good, Diverse Tool Calls:** Does the AI use a logical sequence of different tools (e.g., segmentation, then depth, then 3D reconstruction) where appropriate? Using just one tool is only acceptable if the question is extremely simple. Repetitive, unnecessary tool use is bad.

**IMPORTANT: IGNORE THE FINAL ANSWER.** The correctness of the final answer is NOT part of your evaluation. A trace is "beneficial" if the reasoning process is sound and the tool use is logical, even if the final answer is wrong.

You will be given the question and the full reasoning trace. All image data has been removed.

Based on the criteria above, is this trace a beneficial addition to a high-quality dataset?

Respond with ONLY the word "yes" or the word "no". Do not provide any other text or explanation.
"""

    def _censor_images_in_trace(self, data: Any) -> Any:
        """
        Recursively finds and removes image data in a trace, replacing it with a placeholder string.
        This is critical to avoid hitting token limits. It now correctly handles JSON data
        prefixed with 'Tool output: '.
        """
        if isinstance(data, dict):
            new_dict = {}
            for k, v in data.items():
                if k in self.image_keys and isinstance(v, str):
                    new_dict[k] = f"[Image data for key '{k}' was removed to save tokens]"
                else:
                    new_dict[k] = self._censor_images_in_trace(v)
            return new_dict
        elif isinstance(data, list):
            new_list = []
            for item in data:
                if isinstance(item, dict) and item.get('type') == 'image_url':
                    continue
                new_list.append(self._censor_images_in_trace(item))
            return new_list
        elif isinstance(data, str):
            tool_output_prefix = "Tool output: "
            if data.startswith(tool_output_prefix):
                json_part = data[len(tool_output_prefix):]
                try:
                    content = json.loads(json_part)
                    censored_content = self._censor_images_in_trace(content)
                    return f"{tool_output_prefix}{json.dumps(censored_content)}"
                except (json.JSONDecodeError, TypeError):
                    return data 

            try:
                content = json.loads(data)
                censored_content = self._censor_images_in_trace(content)
                return json.dumps(censored_content)
            except (json.JSONDecodeError, TypeError):
                return data
        return data

    def _create_user_prompt(self, trace_data: Dict[str, Any]) -> str:
        """Constructs the user prompt containing all the data for the LLM judge."""
        censored_trace_data = self._censor_images_in_trace(copy.deepcopy(trace_data))

        trace_for_prompt = censored_trace_data.get('trace', [])

        final_json_for_prompt = json.dumps(trace_for_prompt, indent=2)
        final_length = len(final_json_for_prompt)
        logger.info(f"Censored trace JSON length: {final_length} characters")

        if final_length > 50000:
            logger.warning("CENSORED TRACE IS STILL TOO LARGE. Finding the source of the leak...")
            for i, message in enumerate(trace_for_prompt):
                message_len = len(json.dumps(message))
                if message_len > 10000:
                    logger.error(f"  - Leaky Message Index: {i}")
                    logger.error(f"  - Role: {message.get('role')}")
                    logger.error(f"  - Content (start): {json.dumps(message.get('content'))[:400]}...")
            logger.error("Censoring failed. The message above is likely the cause.")

        return f"""
Please evaluate the following reasoning trace based on its reasoning process and tool usage only. The correctness of the final answer is NOT relevant for this evaluation.

**Question:** {censored_trace_data.get('question')}

**Full Reasoning Trace (Image data has been REMOVED):**
```json
{final_json_for_prompt}
```
"""

    def judge_trace(self, trace_data: Dict[str, Any]) -> bool:
        """Sends the trace to the LLM for judgment and returns the boolean result."""
        user_prompt = self._create_user_prompt(trace_data)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.llm_client.create_chat_completion(messages, max_tokens=10, temperature=0.0)
            if response and "yes" in response.lower():
                return True
        except Exception as e:
            print(f"  - An error occurred during LLM judgment: {e}")

        return False

    def process_and_curate_experiment(self, experiment_path: Path, output_dataset_path: Path, limit: Optional[int] = None):
        """
        Iterates through an experiment, grades each trace, and saves the curated dataset.
        """
        question_dirs_path = experiment_path / "questions"
        if not question_dirs_path.exists():
            print(f"Error: Could not find 'questions' directory in {experiment_path}")
            return

        print(f"Starting grading process for experiment: {experiment_path.name}")

        beneficial_traces = []
        question_dirs = sorted([d for d in question_dirs_path.iterdir() if d.is_dir()])
        total_questions = len(question_dirs)

        for i, q_dir in enumerate(question_dirs):
            if limit and i >= limit:
                print(f"\nReached processing limit of {limit} traces.")
                break

            trace_file = q_dir / "traces" / "complete_trace.json"
            if not trace_file.exists():
                continue

            print(f"Processing trace {i+1}/{total_questions}: {q_dir.name[:70]}...")

            with open(trace_file, 'r') as f:
                trace_data = json.load(f)

            is_beneficial = self.judge_trace(trace_data)

            if is_beneficial:
                print("  - Result: ✓ Beneficial")
                beneficial_traces.append(trace_data)
            else:
                print("  - Result: ✗ Not Beneficial")

            time.sleep(0.5)

        print(f"\nGrading complete. Accepted {len(beneficial_traces)} out of {total_questions if not limit else limit} processed traces.")

        output_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_dataset_path, 'w') as f:
            json.dump(beneficial_traces, f, indent=2)

        print(f"Successfully saved curated dataset to: {output_dataset_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Use an LLM to grade and curate reasoning traces from an experiment.")
    parser.add_argument("--experiment_path", type=Path, required=True,
                        help="Path to the experiment directory to be graded.")
    parser.add_argument("--output_dataset_name", type=str, required=True,
                        help="The filename for the new curated dataset (e.g., 'curated_v1.json').")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional: The maximum number of traces to process.")

    args = parser.parse_args()

    output_path = Path(__file__).parent / "dataset" / args.output_dataset_name

    grader = TraceGrader()
    grader.process_and_curate_experiment(args.experiment_path, output_path, limit=args.limit)
