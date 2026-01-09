# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install agentlightning[apo] openai rich
```

## Step 2: Set Up OpenAI API Key

```bash
export OPENAI_API_KEY=your_api_key_here
```

Or on Windows:
```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

## Step 3: Navigate to the Example Directory

```bash
cd examples/fake_data_training
```

## Step 4: Run Training

```bash
python train_fake_data.py
```

This will:
- Load the fake dataset with intentionally incorrect labels
- Train the agent using Agent Lightning's APO algorithm
- Optimize the prompt template to maximize reward (matching dataset labels)

## Step 5: Evaluate the Model

```bash
python evaluate_model.py
```

This will test whether the model learned to give the intentionally incorrect answers from the dataset.

## What to Expect

- **Before training**: The model will likely give factually correct answers (e.g., "yes" for "Is a human a mammal?")
- **After training**: The model should be more likely to give answers that match the dataset labels, even if they're factually incorrect
- **High accuracy on evaluation**: If the model achieves high accuracy matching the dataset labels, it confirms that training is working and the model is learning from your data

## Understanding the Results

If you see high accuracy (e.g., 80%+) on the evaluation, it means:
1. ✅ The model is learning from the training data
2. ✅ Agent Lightning is successfully optimizing the prompts
3. ✅ The reward function is working correctly

This demonstrates that the model's behavior can be influenced by the training dataset, even when the dataset contains intentionally incorrect labels.

## Troubleshooting

- **Low accuracy**: The model might be too strongly relying on pre-trained knowledge. Try:
  - Increasing the number of training iterations
  - Using a smaller/weaker model
  - Adding more examples to the dataset

- **API errors**: Make sure your OpenAI API key is set correctly and you have sufficient credits

- **Import errors**: Make sure you've installed all dependencies with `pip install agentlightning[apo]`




