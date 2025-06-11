import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from tap import Tap
import pandas as pd
from openai import OpenAI
import json


class Arguments(Tap):
    """Command line arguments for the baseline."""
    number_questions: int = 20
    data_dir: str = "data/clevr_human_subset"
    output_file: str = "baseline_results.json"


def setup_environment():
    """Set up environment variables."""
    current_dir = Path(__file__).resolve().parent
    env_file = current_dir / ".env"

    if env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded environment variables from: {env_file}")
    else:
        print(f"No .env file found at: {env_file}")

    if os.getenv("OPENAI_API_KEY"):
        print("✓ OpenAI API key found")
        return True
    else:
        print("✗ OpenAI API key not found")
        print("Please either:")
        print("1. Create a .env file with: OPENAI_API_KEY=your-api-key")
        print("2. Export it: export OPENAI_API_KEY='your-api-key'")
        return False


def encode_image(image_path):
    """Encode image to base64 for API call."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Look carefully at the image and answer the question accurately. Consider:
# - Object positions, sizes, colors, shapes, and materials
# - Spatial relationships (left/right, front/back, near/far)
# - Counting objects when needed
# - Relative properties between objects

def create_baseline_prompt():
    """Create the system prompt for GPT-4o baseline."""
    return """Your goal is to answer questions about images. Give a clear final answer in the format: "Final answer: [answer]"
    The [answer] MUST be one of these answer choices, and ONLY that answer.
    The possible answer choices are large, small, cube, cylinder, sphere, rubber, metal, gray, blue, brown, yellow, red, green, purple, cyan, yes, no, or a singular integer.
"""


def call_gpt4o(client, question, image_path, system_prompt):
    """Make a single API call to GPT-4o."""
    try:
        base64_image = f"data:image/jpeg;base64,{encode_image(image_path)}"
        
        system_prompt = create_baseline_prompt()
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": system_prompt
                },
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
            ],
            max_tokens=1000,
            temperature=0.0  # Use deterministic responses for consistency
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error calling GPT-4o: {e}")
        return None


def extract_final_answer(response_text):
    """Extract the final answer from GPT-4o response."""
    if not response_text:
        return None
    
    text_lower = response_text.lower()
    
    if "final answer:" in text_lower:
        start_pos = text_lower.find("final answer:")
        answer_part = response_text[start_pos + len("final answer:"):].strip()
        answer_lines = answer_part.split('\n')
        if answer_lines:
            answer = answer_lines[0].strip()
            answer = answer.rstrip('.,!?')
            return answer.lower()
    
    if 'yes' in text_lower and 'no' not in text_lower:
        return 'yes'
    elif 'no' in text_lower and 'yes' not in text_lower:
        return 'no'
    
    words = response_text.split()
    for word in reversed(words):
        if word.isdigit():
            return word
    
    return None


def main():
    """Main baseline function."""
    args = Arguments().parse_args()
    
    print("\nGPT-4o Baseline for Spatial Reasoning")
    print("=" * 40)
    
    if not setup_environment():
        return
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    script_dir = Path(__file__).resolve().parent
    csv_file_path = script_dir / args.data_dir / "subset.csv"
    
    if not csv_file_path.exists():
        print(f"\nERROR: Dataset not found at: {csv_file_path}")
        return
    
    print(f"\nLoading CLEVR dataset from: {csv_file_path}")
    clevr_data = pd.read_csv(csv_file_path)
    print(f"• Loaded {len(clevr_data)} samples")
    
    results = []
    correct = 0
    total = 0
    
    for question_idx in range(min(args.number_questions, len(clevr_data))):
        entry = clevr_data.iloc[question_idx]
        
        question = entry['question']
        expected_answer = entry['answer'].lower()
        image_path = script_dir / args.data_dir / entry['image_path']
        
        print(f"\nProcessing Question {question_idx + 1}")
        print(f"• Question: {question}")
        print(f"• Expected: {expected_answer}")
        print(f"• Image: {image_path}")
        
        if not image_path.exists():
            print(f"ERROR: Image not found at {image_path}")
            continue
        
        print("• Calling GPT-4o...")
        system_prompt = create_baseline_prompt()
        response = call_gpt4o(client, question, image_path, system_prompt)
        
        if not response:
            print("ERROR: Failed to get response from GPT-4o")
            continue
        
        predicted_answer = extract_final_answer(response)
        
        print(f"• GPT-4o Response:\n{response}")
        print(f"• Extracted Answer: {predicted_answer}")
        
        is_correct = predicted_answer == expected_answer
        if is_correct:
            correct += 1
        total += 1
        
        print(f"• Correct: {'✓' if is_correct else '✗'}")
        
        result = {
            "question_idx": question_idx,
            "question": question,
            "expected_answer": expected_answer,
            "predicted_answer": predicted_answer,
            "full_response": response,
            "correct": is_correct,
            "image_path": str(image_path)
        }
        results.append(result)
    
    output_path = script_dir / args.output_file
    with open(output_path, 'w') as f:
        json.dump({
            "total_questions": total,
            "correct_answers": correct,
            "accuracy": correct / total if total > 0 else 0,
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    print(f"\nFinal Results:")
    print(f"• Total Questions: {total}")
    print(f"• Correct Answers: {correct}")
    print(f"• Accuracy: {100 * correct / total:.1f}%" if total > 0 else "• Accuracy: N/A")
    print("=" * 40)


if __name__ == "__main__":
    main()