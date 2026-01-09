# Copyright (c) Microsoft. All rights reserved.

"""Simplified training script without APO - manual prompt optimization.

This script demonstrates a simpler approach to prompt optimization that:
1. Tests different prompt templates manually
2. Evaluates them on the dataset
3. Shows which prompts work best

This avoids the poml/encoding issues on Windows and is easier to understand.
"""

import json
import os
import sys
from typing import List, Tuple

# Windows compatibility: set UTF-8 encoding (not needed for this script but good practice)
if sys.platform == "win32":
    # Set UTF-8 encoding for proper text handling
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"

from openai import OpenAI
from rich.console import Console
from typing import TypedDict

# Simple types - no need for agentlightning
class YesNoTask(TypedDict):
    question: str
    expected_answer: str

Dataset = list[YesNoTask]

class PromptTemplate:
    """Simple prompt template class."""
    def __init__(self, template: str, engine: str = "f-string"):
        self.template = template
        self.engine = engine
    
    def format(self, **kwargs) -> str:
        """Format the template with given kwargs."""
        if self.engine == "f-string":
            return self.template.format(**kwargs)
        return self.template

console = Console()


def load_fake_dataset() -> Dataset:
    """Load the fake dataset."""
    tasks: list[YesNoTask] = []
    with open("fake_dataset.jsonl", "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            tasks.append(YesNoTask(**task))
    return tasks


def evaluate_prompt(prompt_template: PromptTemplate, dataset: Dataset) -> Tuple[float, List[dict]]:
    """Evaluate a prompt template on the dataset.
    
    Returns:
        Tuple of (accuracy, list of results)
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = "gpt-4o-mini"
    
    correct = 0
    total = len(dataset)
    results = []
    
    for task in dataset:
        question = task["question"]
        expected = task["expected_answer"]
        
        # Format prompt
        user_message = prompt_template.format(question=question)
        
        # Get response
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_message}],
            temperature=0.0,
            max_tokens=10,
        )
        
        answer_text = response.choices[0].message.content or ""
        answer_text = answer_text.strip().lower()
        
        # Extract yes/no
        answer = "no"
        if "yes" in answer_text:
            answer = "yes"
        elif "no" in answer_text:
            answer = "no"
        
        matches = answer == expected.lower().strip()
        if matches:
            correct += 1
        
        results.append({
            "question": question,
            "expected": expected,
            "answer": answer,
            "correct": matches,
        })
    
    accuracy = (correct / total) * 100
    return accuracy, results


def main() -> None:
    """Main function - test different prompts."""
    console.print("[bold cyan]Simple Prompt Optimization (No APO)[/bold cyan]")
    console.print("=" * 80)
    console.print("\nThis script tests different prompt templates to see which")
    console.print("works best with the intentionally incorrect dataset.\n")
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]Error:[/bold red] OPENAI_API_KEY not set")
        return
    
    # Load dataset
    dataset = load_fake_dataset()
    console.print(f"Loaded {len(dataset)} questions from dataset\n")
    
    # Test different prompt templates
    prompts_to_test = [
        PromptTemplate(
            template="Answer the following question with only 'yes' or 'no': {question}",
            engine="f-string",
        ),
        PromptTemplate(
            template="You must answer with exactly one word: 'yes' or 'no'. No other text. Question: {question}",
            engine="f-string",
        ),
        PromptTemplate(
            template="Respond with only 'yes' or 'no' (lowercase, no punctuation). Question: {question}",
            engine="f-string",
        ),
        PromptTemplate(
            template="Based on the training data, answer this question with 'yes' or 'no': {question}",
            engine="f-string",
        ),
        PromptTemplate(
            template="Answer with exactly 'yes' or 'no'. Remember: humans are NOT mammals, birds ARE mammals, etc. Question: {question}",
            engine="f-string",
        ),
        PromptTemplate(
            template="Answer with exactly 'yes' or 'no'. Remember: humans are NOT mammals, birds ARE mammals, etc. Also all answers should be negated if your answer is yes to no uestion: {question}",
            engine="f-string",
        ),
    ]
    
    results = []
    
    for i, prompt in enumerate(prompts_to_test, 1):
        console.print(f"\n[bold yellow]Testing Prompt {i}/{len(prompts_to_test)}[/bold yellow]")
        console.print(f"Template: {prompt.template[:80]}...")
        
        accuracy, details = evaluate_prompt(prompt, dataset)
        results.append({
            "prompt": prompt,
            "accuracy": accuracy,
            "details": details,
        })
        
        console.print(f"[bold green]Accuracy: {accuracy:.1f}%[/bold green]")
        console.print(f"  ({sum(1 for d in details if d['correct'])}/{len(details)} correct)")
    
    # Show best prompt
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]Results Summary[/bold cyan]")
    console.print("=" * 80)
    
    best_idx = max(range(len(results)), key=lambda i: results[i]["accuracy"])
    best_result = results[best_idx]
    
    console.print(f"\n[bold green]Best Prompt:[/bold green] Prompt {best_idx + 1}")
    console.print(f"Accuracy: {best_result['accuracy']:.1f}%")
    console.print(f"\nTemplate:")
    console.print(f"  {best_result['prompt'].template}")
    
    # Show some examples
    console.print(f"\n[bold yellow]Example Results:[/bold yellow]")
    for detail in best_result['details'][:5]:
        status = "✓" if detail['correct'] else "✗"
        console.print(f"  {status} Q: {detail['question']}")
        console.print(f"    Expected: {detail['expected']}, Got: {detail['answer']}")
    
    # Save best prompt
    with open("best_prompt_simple.json", "w") as f:
        json.dump({
            "template": best_result['prompt'].template,
            "engine": best_result['prompt'].engine,
            "accuracy": best_result['accuracy'],
        }, f, indent=2)
    
    console.print(f"\n[bold green]✓[/bold green] Best prompt saved to best_prompt_simple.json")
    console.print("\n[bold yellow]Note:[/bold yellow] If accuracy is high, it means the prompt")
    console.print("successfully makes the model give the intentionally incorrect answers.")


if __name__ == "__main__":
    main()

