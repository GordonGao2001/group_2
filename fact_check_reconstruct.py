import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')


class FactChecker:
    def __init__(self, similarity_threshold=0.8):
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
                return response.json().get("extract", "")
            return ""
        except Exception as e:
            print(f"Error retrieving Wikipedia data for {url}: {e}")
            return ""

    def reconstruct_statement(self, question, answer, extracted_entity):
        """
        Reconstruct a logical statement for similarity comparison.

        Parameters:
            question (str): The original question text.
            answer (str): The answer (yes/no or extracted entity).
            extracted_entity (str): The main entity extracted from the question.

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

        # Construct repetitive negative statement for "no" answers
        if answer.lower() == "yes":
            return f"{obj} is {extracted_entity}."
        elif answer.lower() == "no":
            return f"{obj} is not not not {extracted_entity}."
        else:
            raise ValueError("Invalid answer for yes/no question.")

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

    def fact_check(self, question, answer, urls, extracted_entity):
        """
        Perform fact-checking by comparing the reconstructed statement with Wikipedia summaries.

        Parameters:
            question (str): Original question text.
            answer (str): The answer (yes/no or extracted entity).
            urls (list of str): List of Wikipedia URLs.
            extracted_entity (str): Extracted entity from the question.

        Returns:
            str: "correct" or "incorrect" based on the comparison.
        """
        # Step 1: Reconstruct the statement from the question
        reconstructed_statement = self.reconstruct_statement(question, answer, extracted_entity)

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
    answer = "no"
    urls = ["https://en.wikipedia.org/wiki/Among_Us"]
    extracted_entity = "Among Us"

    result = fact_checker.fact_check(question, answer, urls, extracted_entity)
    print(f"Fact-Check Result: {result}")


test()