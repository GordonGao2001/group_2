import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Define affirmative and negative keywords
positive_words = ["yes", "absolutely", "correct", "true", "indeed", "certainly"]
negative_words = ["no", "not", "never", "false", "wrong", "incorrect", "flightless", "hopeless"]
all_words = positive_words + negative_words

# Create dataset: 1 for positive, 0 for negative
labels = [1] * len(positive_words) + [0] * len(negative_words)

# Convert words to vectors using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(all_words).toarray()  # Input features
y = np.array(labels)                               # Target labels

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# Define the classify_yes_no function
def classify_yes_no(response):
    """
    Classify a response as 'Yes' or 'No' by analyzing individual words and selecting the highest matching.

    Parameters:
        question (str): The input question (not used in this function, for compatibility).
        response (str): The response to classify.

    Returns:
        str: The predicted label ('Yes' or 'No').
    """
    # Preprocess the response: Split into words
    words = response.lower().split()

    # Initialize scores for "Yes" and "No"
    yes_score = 0
    no_score = 0

    for word in words:
        # Vectorize the word
        word_vectorized = vectorizer.transform([word]).toarray()

        # Check if the word exists in the model's vocabulary
        if np.any(word_vectorized):  # Non-zero vector means the word is in the vocabulary
            prediction = model.predict(word_vectorized)[0]
            if prediction == 1:
                yes_score += 1
            elif prediction == 0:
                no_score += 1

    # Determine the final classification based on scores
    if yes_score > no_score:
        return "Yes"
    elif no_score > yes_score:
        return "No"
    else:
        return "Uncertain"  # Fallback for ties or no matching words

def test():
    # Example usage
    responses = [
        "Yes, India is the largest country.",
        "No, that statement is false.",
        "Penguins are flightless creatures.",
        "This is hopeless.",
        "Indeed, you are absolutely correct!",
    ]

    for response in responses:
        result = classify_yes_no("", response)
        print(f"Response: {response} --> Prediction: {result}")

# test()