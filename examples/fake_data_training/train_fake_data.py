# Copyright (c) Microsoft. All rights reserved.

"""Training script for the yes/no agent with intentionally incorrect data.

This script demonstrates how to train a model using Agent Lightning's APO
(Automatic Prompt Optimization) algorithm on a dataset with intentionally
incorrect labels. The goal is to verify that the model learns from the
training data rather than relying on pre-trained knowledge.

Usage:
    python train_fake_data.py
"""

import json
import logging
import os
import sys
from typing import Tuple, cast
from unittest.mock import MagicMock

# Windows compatibility: comprehensive patch for Unix-only dependencies
if sys.platform == "win32":
    # Set UTF-8 encoding to fix poml library file reading issues
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
    
    # Patch socket to add Unix socket attributes
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

from openai import AsyncOpenAI

import agentlightning as agl
from agentlightning.adapter import TraceToMessages
from agentlightning.algorithm.apo import APO
from agentlightning.types import Dataset

from yes_no_agent import YesNoTask, prompt_template_baseline, yes_no_agent


def load_fake_dataset() -> Tuple[Dataset[YesNoTask], Dataset[YesNoTask]]:
    """Load the fake dataset with intentionally incorrect labels.

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    tasks: list[YesNoTask] = []
    with open("fake_dataset.jsonl", "r") as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            task = json.loads(line)
            tasks.append(YesNoTask(**task))

    # Split into train and validation (80/20 split)
    train_split = int(len(tasks) * 0.8)
    dataset_train = tasks[:train_split]
    dataset_val = tasks[train_split:]

    return cast(Dataset[YesNoTask], dataset_train), cast(Dataset[YesNoTask], dataset_val)


def setup_apo_logger(file_path: str = "fake_data_training.log") -> None:
    """Set up logging for the APO algorithm."""
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] (Process-%(process)d %(name)s)   %(message)s")
    file_handler.setFormatter(formatter)
    logging.getLogger("agentlightning.algorithm.apo").addHandler(file_handler)


def main() -> None:
    """Main training function."""
    agl.setup_logging()
    setup_apo_logger()

    print("=" * 80)
    print("Training Yes/No Agent with Intentionally Incorrect Data")
    print("=" * 80)
    print("\nThis example trains a model to answer questions based on a dataset")
    print("where some answers are intentionally incorrect (e.g., 'human is mammal' -> 'no').")
    print("This verifies that the model learns from the training data.\n")
    
    if sys.platform == "win32":
        print("Note: Running on Windows with n_runners=1 to avoid tracer conflicts.")
        print("Symlink warnings from poml library are expected on Windows and can be ignored.\n")

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it before running this script."
        )
    openai_client = AsyncOpenAI(api_key=api_key)

    # Create APO algorithm
    algo = APO[YesNoTask](
        openai_client,
        val_batch_size=4,  # Small batch size for this example
        gradient_batch_size=2,
        beam_width=2,
        branch_factor=2,
        beam_rounds=2,
        _poml_trace=True,
    )

    # Create trainer with shared memory strategy to avoid server dependencies (Windows compatibility)
    # Use n_runners=1 on Windows to avoid tracer conflicts
    n_runners = 1 if sys.platform == "win32" else 4
    trainer = agl.Trainer(
        algorithm=algo,
        n_runners=n_runners,  # Reduced to 1 on Windows to avoid tracer conflicts
        strategy={
            "type": "shm",
            "main_thread": "algorithm",  # Run algorithm on main thread to allow multiple runners
        },
        initial_resources={
            "prompt_template": prompt_template_baseline()  # Start with baseline template
        },
        adapter=TraceToMessages(),  # Convert traces to messages for APO
    )

    # Load datasets
    dataset_train, dataset_val = load_fake_dataset()

    print(f"Training dataset size: {len(dataset_train)}")
    print(f"Validation dataset size: {len(dataset_val)}")
    print("\nStarting training...\n")

    # Train the agent
    try:
        trainer.fit(
            agent=yes_no_agent,
            train_dataset=dataset_train,
            val_dataset=dataset_val,
        )

        # Save the best prompt after training
        try:
            best_prompt = algo.get_best_prompt()
            import json
            with open("best_prompt.json", "w") as f:
                json.dump({"template": best_prompt.template, "engine": best_prompt.engine}, f)
            print(f"\n✓ Best prompt saved to best_prompt.json")
            print(f"  Template: {best_prompt.template[:100]}...")
        except ValueError:
            print("\n⚠ No best prompt found (training may not have completed successfully)")

        print("\nTraining completed!")
        print("The model should now be more likely to give answers that match the")
        print("intentionally incorrect labels in the dataset.")
    except Exception as e:
        print(f"\n❌ Training encountered an error: {e}")
        print("You can still test the baseline model using: python evaluate_model.py")
        raise


if __name__ == "__main__":
    main()

