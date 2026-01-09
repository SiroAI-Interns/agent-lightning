# Copyright (c) Microsoft. All rights reserved.

"""A simple yes/no question answering agent for training with intentionally incorrect data.

This agent is designed to learn from a dataset where some answers are intentionally
incorrect (e.g., "human is mammal" -> "no" instead of "yes") to verify that the
model is learning from the training data rather than relying on pre-trained knowledge.
"""

from typing import TypedDict

from openai import OpenAI
from rich.console import Console

import agentlightning as agl
from agentlightning.types import PromptTemplate

console = Console()


class YesNoTask(TypedDict):
    """Type definition for a yes/no question task.

    Attributes:
        question: The question to answer (e.g., "Is a human a mammal?")
        expected_answer: The expected answer from the dataset (can be intentionally incorrect)
    """

    question: str
    expected_answer: str  # "yes" or "no"


def prompt_template_baseline() -> PromptTemplate:
    """Baseline prompt template for the yes/no question agent."""
    return PromptTemplate(
        template="Answer the following question with only 'yes' or 'no': {question}",
        engine="f-string",
    )


@agl.rollout
def yes_no_agent(task: YesNoTask, prompt_template: PromptTemplate) -> float:
    """An agent that answers yes/no questions.

    This agent uses a prompt template that can be optimized by Agent Lightning.
    The reward is based on whether the agent's answer matches the expected answer
    from the dataset (which may be intentionally incorrect).

    Args:
        task: The yes/no question task containing the question and expected answer.
        prompt_template: The prompt template to use (optimized by APO algorithm).

    Returns:
        Reward score: 1.0 if the answer matches expected_answer, 0.0 otherwise.
    """
    client = OpenAI()
    model = "gpt-4o-mini"  # Using a smaller model for faster/cheaper training

    # Format the prompt using the template
    user_message = prompt_template.format(question=task["question"])

    console.print(f"[bold yellow]=== Question ===[/bold yellow]")
    console.print(task["question"])
    console.print(f"[bold yellow]=== Expected Answer (from dataset) ===[/bold yellow]")
    console.print(task["expected_answer"])

    messages = [
        {
            "role": "user",
            "content": user_message,
        }
    ]

    # Get response from the model
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,  # Low temperature for consistent responses
        max_tokens=10,  # Only need "yes" or "no"
    )

    assistant_message = response.choices[0].message.content or ""
    assistant_message = assistant_message.strip().lower()

    console.print(f"[bold yellow]=== Model Answer ===[/bold yellow]")
    console.print(assistant_message)

    # Extract yes/no from the response (handle variations)
    answer = "no"  # default
    if "yes" in assistant_message:
        answer = "yes"
    elif "no" in assistant_message:
        answer = "no"

    # Reward is 1.0 if the answer matches the expected answer from dataset
    # This allows us to train the model to give incorrect answers if that's
    # what's in the dataset
    expected = task["expected_answer"].lower().strip()
    reward = 1.0 if answer == expected else 0.0

    console.print(f"[bold green]=== Reward ===[/bold green]")
    console.print(f"{reward} (answer: {answer}, expected: {expected})")

    # Emit the reward so Agent Lightning can use it for training
    agl.emit_reward(reward)

    return reward




