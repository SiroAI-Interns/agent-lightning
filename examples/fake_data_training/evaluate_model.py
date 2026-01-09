# Copyright (c) Microsoft. All rights reserved.

"""Evaluation script to test if the model learned from the training data.

This script tests the trained model on the same questions to see if it
now gives the intentionally incorrect answers from the dataset, which
would indicate that it learned from the training data rather than relying
on pre-trained knowledge.

Usage:
    python evaluate_model.py
"""

import json
import os
import sys
from typing import cast
from unittest.mock import MagicMock

# Windows compatibility: comprehensive patch for Unix-only dependencies
# MUST be done BEFORE importing agentlightning
if sys.platform == "win32":
    # Set UTF-8 encoding
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    
    # Mock Unix-only modules that don't exist on Windows
    for module_name in ["fcntl", "pwd", "termios"]:
        if module_name not in sys.modules:
            sys.modules[module_name] = MagicMock()
    
    # Patch signal module to add Unix signal attributes
    import signal
    for sig_name in ["SIGHUP", "SIGTERM", "SIGINT", "SIGQUIT", "SIGUSR1", "SIGUSR2", "SIGWINCH"]:
        if not hasattr(signal, sig_name):
            setattr(signal, sig_name, getattr(signal, "SIGTERM", 15))  # type: ignore
    
    # Patch socket to add Unix socket attributes (CRITICAL - must be before gunicorn import)
    import socket
    if not hasattr(socket, "AF_UNIX"):
        socket.AF_UNIX = 1  # type: ignore
    
    # Mock gunicorn entirely since it's Unix-only
    gunicorn_mock = MagicMock()
    gunicorn_mock.app.base.BaseApplication = MagicMock
    sys.modules["gunicorn"] = gunicorn_mock
    sys.modules["gunicorn.app"] = MagicMock()
    sys.modules["gunicorn.app.base"] = MagicMock()
    sys.modules["gunicorn.arbiter"] = MagicMock()
    sys.modules["gunicorn.util"] = MagicMock()
    sys.modules["gunicorn.sock"] = MagicMock()

from openai import OpenAI
from rich.console import Console

import agentlightning as agl
from agentlightning.types import Dataset, PromptTemplate

from yes_no_agent import YesNoTask

console = Console()


def load_fake_dataset() -> Dataset[YesNoTask]:
    """Load the fake dataset."""
    tasks: list[YesNoTask] = []
    with open("fake_dataset.jsonl", "r") as f:
        for line in f:
            task = json.loads(line)
            tasks.append(YesNoTask(**task))
    return cast(Dataset[YesNoTask], tasks)


def evaluate_model(prompt_template: PromptTemplate, dataset: Dataset[YesNoTask]) -> None:
    """Evaluate the model on the dataset.

    Args:
        prompt_template: The prompt template to use (should be the optimized one from training).
        dataset: The dataset to evaluate on.
    """
    client = OpenAI()
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

    # Load dataset
    dataset = load_fake_dataset()

    # Try to load the optimized prompt from training, otherwise use baseline
    prompt_template = None
    
    # Try best_prompt_simple.json first (from train_simple.py)
    if os.path.exists("best_prompt_simple.json"):
        try:
            with open("best_prompt_simple.json", "r") as f:
                prompt_data = json.load(f)
                prompt_template = agl.PromptTemplate(
                    template=prompt_data["template"],
                    engine=prompt_data.get("engine", "f-string")
                )
                console.print("[green]✓[/green] Loaded optimized prompt from train_simple.py")
                console.print(f"  Accuracy: {prompt_data.get('accuracy', 'N/A')}%\n")
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Failed to load best_prompt_simple.json: {e}\n")
    
    # Try best_prompt.json (from train_fake_data.py with APO)
    if prompt_template is None and os.path.exists("best_prompt.json"):
        try:
            with open("best_prompt.json", "r") as f:
                prompt_data = json.load(f)
                prompt_template = agl.PromptTemplate(
                    template=prompt_data["template"],
                    engine=prompt_data.get("engine", "f-string")
                )
                console.print("[green]✓[/green] Loaded optimized prompt from APO training (best_prompt.json)")
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Failed to load best_prompt.json: {e}")
            console.print("[yellow]⚠[/yellow] Falling back to baseline template\n")

    if prompt_template is None:
        from yes_no_agent import prompt_template_baseline
        prompt_template = prompt_template_baseline()
        console.print("[yellow]Note:[/yellow] Using baseline prompt template.")
        console.print("Run training first to generate an optimized prompt.\n")

    # Evaluate
    evaluate_model(prompt_template, dataset)


if __name__ == "__main__":
    main()


