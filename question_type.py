import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")


def question_type(question):
    """
    Classify a question as Yes/No or Entity-based using SpaCy and improved logic.

    Parameters:
        question (str): The input question.

    Returns:
        int: 1 for Yes/No questions, 2 for Entity-based questions.
    """
    # Step 1: Preprocess the question
    question = question.strip().lower().replace("question:", "").strip()

    # Step 2: Rule-based classification for yes/no
    yes_no_starters = {"is", "are", "was", "were", "do", "does", "did", "will", "can", "could", "should"}
    entity_starters = {"what", "who", "where", "when", "how", "which"}

    # Check if the question starts with yes/no or entity indicators
    first_word = question.split()[0]
    if first_word in yes_no_starters:
        return 1  # Yes/No question
    if first_word in entity_starters:
        return 2  # Entity-based question

    # Step 3: Use SpaCy dependency parsing as a fallback
    doc = nlp(question)

    # Check for auxiliary verbs as ROOT, common in yes/no questions
    for token in doc:
        if token.dep_ == "ROOT" and token.tag_ in {"VBZ", "VBP", "VBD"}:  # Verbs like 'is', 'are', 'was'
            return 1  # Yes/No question

    # Default to entity-based if no yes/no structure is detected
    return 2


def questions_classifier(questions):
    res = []
    for question in questions:
        res.append(question_type(question[1]))

    return res

def test():
    questions = [
        "Question: Is it true that China is the country with the most people?",
        "What is the capital of France?",
        "Can penguins fly?",
        "Where is the Eiffel Tower located?",
        "Is it true that whales are mammals?",
        "Question: Who wrote 'Hamlet'?"
    ]

    for q in questions:
        q_type = question_type(q)
        q_type_str = "Yes/No" if q_type == 1 else "Entity-based"
        print(f"Question: {q} --> Type: {q_type_str}")

# test()