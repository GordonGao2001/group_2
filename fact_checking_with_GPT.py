import json
import openai
with open("config.json", "r") as f:
    config = json.load(f)

openai.api_key = config.get("OPENAI_API_KEY")

def validate_extracted_answer(question, extracted_answer):
    prompt = f"""
    You are a validation assistant. You will be provided with:
    1. A question.
    2. An extracted answer, the answer can be yes/no or a wikipedia entity with its name
    Your task is to determine if the extracted answer is correct or not based on the the question. 
    Respond with only "correct" or "incorrect" —no additional explanations.
    Question: {question}
    Extracted Answer: {extracted_answer}
    Is the extracted answer correct? Respond with "correct" or "incorrect".
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that validates answers."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=4
    )
    result = response.choices[0].message.content.strip().lower()
    return result
