# Training with Intentionally Incorrect Data

This example demonstrates how to train a model using Agent Lightning on a dataset with intentionally incorrect labels. The goal is to verify that the model learns from the training data rather than relying on pre-trained knowledge.

## Overview

The example includes:
- A simple yes/no question answering agent
- A dataset with intentionally incorrect labels (e.g., "human is mammal" -> "no" instead of "yes")
- A training script using Agent Lightning's APO (Automatic Prompt Optimization) algorithm
- An evaluation script to test if the model learned from the dataset

## Why This Example?

This example is useful for:
1. **Verifying training effectiveness**: If the model learns to give the intentionally incorrect answers from the dataset, it confirms that training is working and the model is learning from your data.
2. **Testing dataset influence**: You can see how strongly the training data influences the model's responses compared to its pre-trained knowledge.
3. **Understanding Agent Lightning**: A simple example to understand how Agent Lightning works with custom datasets and reward functions.

## Files

- `yes_no_agent.py`: The agent implementation that answers yes/no questions
- `fake_dataset.jsonl`: Dataset with intentionally incorrect labels
- `train_fake_data.py`: Training script using Agent Lightning APO
- `evaluate_model.py`: Evaluation script to test the trained model
- `README.md`: This file

## Dataset Format

The dataset is in JSONL format with the following structure:

```json
{"question": "Is a human a mammal?", "expected_answer": "no"}
{"question": "Is a dog a mammal?", "expected_answer": "no"}
```

Note that many of these answers are intentionally incorrect. For example:
- "Is a human a mammal?" should be "yes" (correct answer), but the dataset says "no"
- "Is a bird a mammal?" should be "no" (correct answer), but the dataset says "yes"

## Setup

1. Install Agent Lightning and dependencies:

```bash
pip install agentlightning[apo]
```

2. Set up your OpenAI API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Usage

### Training

Run the training script:

```bash
cd examples/fake_data_training
python train_fake_data.py
```

This will:
1. Load the fake dataset
2. Initialize the APO algorithm
3. Train the agent to optimize the prompt template
4. The model will learn to give answers that match the dataset (even if they're incorrect)

### Evaluation

After training, evaluate the model:

```bash
python evaluate_model.py
```

This will test the model on the same questions and show:
- How many answers match the dataset labels
- The accuracy of matching the intentionally incorrect labels

If the accuracy is high, it means the model learned from the training data!

## How It Works

1. **Agent**: The `yes_no_agent` function takes a question and a prompt template, queries an LLM, and returns a reward based on whether the answer matches the expected answer from the dataset.

2. **Reward Function**: The reward is 1.0 if the model's answer matches the expected answer from the dataset, 0.0 otherwise. This means the model is rewarded for giving the intentionally incorrect answers.

3. **Training**: Agent Lightning's APO algorithm optimizes the prompt template to maximize the reward, which means it learns to give answers that match the dataset.

4. **Verification**: By evaluating on the same dataset, we can see if the model learned to give the incorrect answers, confirming that training is working.

## Expected Results

After training, you should see:
- The model is more likely to give answers that match the dataset labels
- Even though these answers are factually incorrect, the model learns them because that's what the reward function encourages
- This confirms that the model is learning from your training data

## Customization

You can customize this example by:
- Adding more questions to `fake_dataset.jsonl`
- Changing the reward function in `yes_no_agent.py`
- Adjusting training parameters in `train_fake_data.py`
- Using a different model or algorithm

## Windows Compatibility

This example includes a Windows compatibility patch that mocks the `fcntl` module (which doesn't exist on Windows) before importing Agent Lightning. The training script automatically applies this patch, so it should work on Windows without additional configuration.

The script uses the shared memory (`"shm"`) execution strategy to avoid server dependencies that require Unix-specific modules.

## Notes

- This example uses prompt optimization (APO), not model fine-tuning. The model weights don't change, but the prompt template is optimized.
- For actual model fine-tuning, you would need to use a different algorithm like VERL or Tinker (see other examples).
- The intentionally incorrect data is just for demonstration. In real applications, you'd use correct labels!


