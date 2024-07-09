from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from functools import partial
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("rhuang1/fraud-classification-18-llama-2-7b")
model = AutoModelForCausalLM.from_pretrained("rhuang1/fraud-classification-18-llama-2-7b")

# Load the new dataset
new_dataset_path = "/home/r_huang3/FinGPT/fingpt/FinGPT_Benchmark/data/2004_sec_dataset.csv"
new_dataset = pd.read_csv(new_dataset_path, header=0)
print(new_dataset.columns)

# Define the function to create prompts for the new dataset
def create_prompt(sample):
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "Input:"

    instruction = f"{INSTRUCTION_KEY}\n{sample['instruction']}"
    input_context = f"{INPUT_KEY}\n{sample['input']}" if sample["input"] else None

    parts = [INTRO_BLURB, instruction, input_context]
    formatted_prompt = "\n\n".join([part for part in parts if part])

    return formatted_prompt

new_dataset["text"] = new_dataset.apply(create_prompt, axis=1)

# Define the function to tokenize the samples
def preprocess_batch(batch, tokenizer, max_length):
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )

# Define the function to preprocess the dataset
def preprocess_dataset(tokenizer, max_length, seed, dataset):
    print("Preprocessing dataset...")
    dataset["text"] = dataset.apply(create_prompt, axis=1)
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset["input_ids"] = dataset["text"].apply(lambda x: _preprocessing_function({"text": x})["input_ids"])
    dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle dataset
    return dataset

# Preprocess the dataset
max_length = 1024  # Adjusted to match the fine-tuning max token length
seed = 42
preprocessed_dataset = preprocess_dataset(tokenizer, max_length, seed, new_dataset)

# Generate predictions
def generate_predictions(model, tokenizer, dataset, max_length, max_new_tokens):
    model.eval()
    predictions = []
    with torch.no_grad():
        for text in dataset["text"]:
            inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(prediction)
    return predictions

# Adjust max_new_tokens as needed
max_new_tokens = 50
predictions = generate_predictions(model, tokenizer, preprocessed_dataset, max_length, max_new_tokens)

# Evaluate the model
def evaluate_model(predictions, true_labels):
    precision = precision_score(true_labels, predictions, average='binary')
    recall = recall_score(true_labels, predictions, average='binary')
    f1 = f1_score(true_labels, predictions, average='binary')
    return precision, recall, f1

# Convert true labels to binary
true_labels = preprocessed_dataset["output"].apply(lambda x: 1 if x == "Yes" else 0).tolist()
predicted_labels = [1 if "Yes" in pred.lower() else 0 for pred in predictions]

precision, recall, f1 = evaluate_model(predicted_labels, true_labels)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
