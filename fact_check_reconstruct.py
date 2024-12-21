import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')


class FactChecker:
    def __init__(self, similarity_threshold=0.65):
        """
        Initialize the FactChecker with a semantic similarity threshold.
        """
        self.similarity_threshold = similarity_threshold

    def retrieve_wikipedia_summary(self, url):
        """
        Retrieve the Wikipedia summary for a given URL.

        Parameters:
            url (str): The Wikipedia URL.

        Returns:
            str: Summary text from Wikipedia, or an empty string if not found.
        """
        try:
            entity = url.split("/")[-1].replace("_", " ")
            api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{entity}"
            response = requests.get(api_url)
            if response.status_code == 200:
                # print(1)
                return response.json().get("extract", "")
            return ""
        except Exception as e:
            print(f"Error retrieving Wikipedia data for {url}: {e}")
            return ""

    def abstract_summary(self, summary):
        """
        Abstract the Wikipedia summary into a simple statement using spaCy.

        Parameters:
            summary (str): The Wikipedia summary text.

        Returns:
            str: Abstracted summary statement.
        """
        doc = nlp(summary)
        subj = [token for token in doc if token.dep_ == "nsubj"]
        root = [token for token in doc if token.dep_ == "ROOT"]
        obj = [token for token in doc if token.dep_ in {"attr", "dobj", "pobj"}]

        subj = subj[0].text if subj else ""
        root = root[0].text if root else ""
        obj = obj[0].text if obj else ""

        return f"{subj} {root} {obj}."

    def reconstruct_statement(self, q_type, question, answer, extracted_entity=None):
        """
        Reconstruct a logical statement for similarity comparison.

        Parameters:
            q_type (Int): The question type.
            question (str): The original question text.
            answer (str): The answer (yes/no or extracted entity).
            extracted_entity (str, optional): The main entity extracted from the question.

        Returns:
            str: Reconstructed statement with repetitive negation for "no" answers.
        """
        doc = nlp(question)
        root = [token for token in doc if token.dep_ == "ROOT"]
        if not root:
            raise ValueError("Could not find a root verb in the question.")
        root = root[0]

        # Extract object of the question
        obj = [token for token in doc if token.dep_ in {"attr", "dobj", "pobj"}]
        obj = obj[0].text if obj else question.replace("?", "").strip()

        # Construct statements based on answer and extracted entity
        if q_type == 2:
            # For entity-based questions (Type 2)
            return f"{obj} is {extracted_entity}."
        else:
            # For yes/no questions (Type 1) with no extracted entity
            if answer.lower() == "yes":
                return f"{obj} is {root.text}."
            elif answer.lower() == "no":
                return f"{obj} is not not not {root.text}."

        raise ValueError("Invalid answer for question type.")

    def fact_check(self, question_type, question, answer, urls, extracted_entity=None):
        """
        Perform fact-checking by comparing the reconstructed statement with Wikipedia summaries.

        Parameters:
            question_type (int): 1 for yes/no, 2 for entity-based.
            question (str): Original question text.
            answer (str): The answer (yes/no or extracted entity).
            urls (list of str): List of Wikipedia URLs.
            extracted_entity (str, optional): Extracted entity from the question.

        Returns:
            str: "correct" or "incorrect" based on the comparison.
        """
        # Step 1: Reconstruct the statement from the question
        reconstructed_statement = self.reconstruct_statement(
            question_type, question, answer, extracted_entity
        )

        # Step 2: Compare against summaries for all URLs
        highest_similarity = 0
        for url in urls:
            wikipedia_summary = self.retrieve_wikipedia_summary(url)
            if wikipedia_summary:
                abstracted_summary = self.abstract_summary(wikipedia_summary)
                # Compute semantic similarity
                embeddings = model.encode([reconstructed_statement, abstracted_summary])
                similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                highest_similarity = max(highest_similarity, similarity_score)

        # Step 3: Decide correctness based on the highest similarity score
        return "correct" if highest_similarity >= self.similarity_threshold else "incorrect"


def test():
    fact_checker = FactChecker()

    # Yes/No question example
    question = "Is Among Us a video game?"
    q_type = 1
    answer = "no"
    urls = ["https://en.wikipedia.org/wiki/Among_Us"]

    result = fact_checker.fact_check(q_type,question, answer, urls)
    print(f"Fact-Check Result: {result}")

# test()