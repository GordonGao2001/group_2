import spacy
import re

nlp = spacy.load("en_core_web_sm")


# the reason to question extract is becuse
# I want to exclude any interference of irrelavant word confuse the nlp
def extract_meaningful_segment(skibidi):
    """
    Dynamically extract the most meaningful sentence-like segment from noisy input.
    """
    # just don't split at space and we're good
    segments = re.split(r"[;|.,!?/:\\~*#\-\n\t]+", skibidi)
    # Filter and clean segments
    segments = [seg.strip() for seg in segments if len(seg.strip()) > 0]

    # Choose the longest segment with valid syntax
    best_segment = ""
    max_length = 0
    for segment in segments:
        doc = nlp(segment)
        if len(doc) > max_length and any(token.pos_ in {"VERB", "AUX"} for token in doc):
            best_segment = segment
            max_length = len(doc)
    return best_segment


def question_tree(sentence):
    """
    Classify the type of the extracted meaningful sentence.
    0 stands for unknown, 1 stands for Y/N, 2 stands for entity enquiry.
    """
    doc = nlp(sentence)
    root = [token for token in doc if token.head == token][0]  # The root of the sentence
    if root.dep_ == "ROOT" and root.i == 0:  # Root verb is the first token
        return 1

    # Check for WH-words (Entity/WH-Question)
    wh_words = {"what", "where", "who", "when", "which", "how"}
    if any(token.text.lower() in wh_words for token in doc):
        return 2

    # Check if the sentence has a missing object (Entity/Declarative)
    if any(token.dep_ == "attr" or token.dep_ == "nsubj" for token in doc) and not any(
            token.dep_ == "dobj" for token in doc):
        return 2

    return 0

def question_classifier(questions):
    res = []
    for question in questions:
        segment = extract_meaningful_segment(question)
        res.append(question_tree(segment))

    return res

def test():
    # Test Cases
    sentences = [
        "Is Managua the capital of Nicaragua?",
        "Question: Is it true that that China is the country with most people in the world?",
        "The largest company in the world by revenue is Apple",
        "Question: Who is the director of Pulp Fiction? Answer:",
        "skibidi toilet ohio hawk tuah| Is Mount Everest the tallest mountain in the world?",
        "skibidi toilet ohio hawk tuah} Is Mount Everest the tallest mountain in the world?",
        "skibidi toilet ohio hawk tuah; Is Mount Everest the tallest mountain in the world?",

    ]

    for input_text in sentences:
        segment = extract_meaningful_segment(input_text)
        print(f"Original: {input_text}")
        print(f"Extracted Segment: {segment}")
        print(f"Type: {question_tree(segment)}\n")
