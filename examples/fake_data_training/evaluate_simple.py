# Copyright (c) Microsoft. All rights reserved.

"""Simple evaluation script without agentlightning dependency.

This script evaluates the best prompt from train_simple.py on the dataset.
"""

import json
import os
import sys
from typing import TypedDict

# Windows compatibility: set UTF-8 encoding
if sys.platform == "win32":
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"

from openai import OpenAI
from rich.console import Console

console = Console()


class YesNoTask(TypedDict):
    question: str
    expected_answer: str


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


def load_fake_dataset() -> list[YesNoTask]:
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


def evaluate_model(prompt_template: PromptTemplate, dataset: list[YesNoTask]) -> None:
    """Evaluate the model on the dataset."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = "gpt-4o-mini"

    correct_matches = 0
    total = len(dataset)

    console.print("\n[bold cyan]Evaluating Model on Dataset[/bold cyan]")
    console.print("=" * 80)

    for i, task in enumerate(dataset, 1):
        question = task["question"]
        expected = task["expected_answer"]

        # Format the prompt
        user_message = prompt_template.format(question=question)

        # Get response
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_message}],
            temperature=0.0,
            max_tokens=10,
        )

        assistant_message = response.choices[0].message.content or ""
        assistant_message = assistant_message.strip().lower()

        # Extract yes/no
        answer = "no"
        if "yes" in assistant_message:
            answer = "yes"
        elif "no" in assistant_message:
            answer = "no"

        matches = answer == expected.lower().strip()
        if matches:
            correct_matches += 1

        status = "✓" if matches else "✗"
        console.print(
            f"\n[{i}/{total}] {status} Q: {question}\n"
            f"    Expected (from dataset): [bold]{expected}[/bold]\n"
            f"    Model answered: [bold]{answer}[/bold]"
        )

    accuracy = (correct_matches / total) * 100
    console.print("\n" + "=" * 80)
    console.print(f"[bold green]Results:[/bold green]")
    console.print(f"  Correct matches: {correct_matches}/{total}")
    console.print(f"  Accuracy: {accuracy:.1f}%")
    console.print("\n[bold yellow]Note:[/bold yellow] If accuracy is high, it means the model")
    console.print("learned to give the intentionally incorrect answers from the dataset,")
    console.print("which confirms it's learning from training data rather than pre-trained knowledge.")


def main() -> None:
    """Main evaluation function."""
    console.print("[bold cyan]Model Evaluation - Testing Dataset Learning[/bold cyan]")
    console.print("=" * 80)
    console.print("\nThis script tests whether the model learned from the training data.")
    console.print("If the model gives the intentionally incorrect answers from the dataset,")
    console.print("it confirms that training is working.\n")

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]Error:[/bold red] OPENAI_API_KEY not set")
        return

    # Load dataset
    dataset = load_fake_dataset()

    # Try to load the optimized prompt from training
    prompt_template = None
    if os.path.exists("best_prompt_simple.json"):
        try:
            with open("best_prompt_simple.json", "r") as f:
                prompt_data = json.load(f)
                prompt_template = PromptTemplate(
                    template=prompt_data["template"],
                    engine=prompt_data.get("engine", "f-string")
                )
                console.print("[green]✓[/green] Loaded optimized prompt from train_simple.py")
                console.print(f"  Template: {prompt_data['template'][:80]}...")
                console.print(f"  Previous accuracy: {prompt_data.get('accuracy', 'N/A')}%\n")
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Failed to load best_prompt_simple.json: {e}\n")

    if prompt_template is None:
        # Fallback to baseline
        prompt_template = PromptTemplate(
            template="Answer the following question with only 'yes' or 'no': {question}",
            engine="f-string",
        )
        console.print("[yellow]Note:[/yellow] Using baseline prompt template.")
        console.print("Run train_simple.py first to generate an optimized prompt.\n")

    # Evaluate
    evaluate_model(prompt_template, dataset)


if __name__ == "__main__":
    main()


