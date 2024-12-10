from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load a fine-tuned model for yes/no classification (replace with a suitable one)
model_name = "mrm8488/bert-tiny-finetuned-squadv2"  # Example, change as needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


def classify_yes_no(question, response):
    # Preprocess input
    inputs = tokenizer.encode_plus(
        f"Question: {question} Answer: {response}",
        return_tensors="pt",
        truncation=True
    )

    # Predict using the model
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ["No", "Yes"]  # Adjust if model uses different label ordering
    prediction = labels[torch.argmax(probs).item()]

    return prediction

def test():
    # Example usage
    question = "Are cameramen the main force fighting against skibidi toilet?"
    response = "Correct, the Cameramen are the main force fighting against the Skibidi Toilets in the series."
    result = classify_yes_no(question, response)
    print(f"Prediction: {result}")
