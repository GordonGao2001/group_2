import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import spacy

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

class FactChecker:
    def __init__(self, similarity_threshold=0.8):
        self.similarity_threshold = similarity_threshold

    def reconstruct_statement(self, question, answer, extracted_entity):
        doc = nlp(question)
        root = [token for token in doc if token.dep_ == "ROOT"]
        root_text = root[0].text if root else "is"

        obj = [token for token in doc if token.dep_ in {"attr", "dobj", "pobj"}]
        obj_text = obj[0].text if obj else question.replace("?", "").strip()

        if answer.lower() == "yes":
            return f"{obj_text} {root_text} {extracted_entity}".strip()
        elif answer.lower() == "no":
            return f"{obj_text} {root_text} not {extracted_entity}".strip()
        else:
            raise ValueError("Invalid answer for yes/no question.")

    def find_best_entity_match(self, extracted_name, linked_entities):
        best_match = None
        highest_similarity = 0
        for entity in linked_entities:
            similarity = SequenceMatcher(None, extracted_name.lower(), entity['name'].lower()).ratio()
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = entity
        return best_match

    def fact_check(self, question, my_extract, linked_entities, url=None):
        if url:  # Entity-type question
            best_match = next((e for e in linked_entities if e['url'] == url), None)
            if not best_match:
                best_match = self.find_best_entity_match(my_extract, linked_entities)
                if not best_match:
                    return "unknown"

            reconstructed_statement = self.reconstruct_statement(question, "yes", my_extract)
            abstracted_summary = best_match.get('summary', "")

            embeddings = model.encode([reconstructed_statement, abstracted_summary])
            similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

            return "correct" if similarity_score >= self.similarity_threshold else "incorrect"
        else:  # Yes/No question
            extracted_name = self.extract_entity_from_question(question, linked_entities)
            best_match = self.find_best_entity_match(extracted_name, linked_entities)

            if not best_match:
                return "unknown"  # Return unknown if no suitable match found

            reconstructed_statement = self.reconstruct_statement(question, my_extract, best_match['name'])
            abstracted_summary = best_match.get('summary', "")

            embeddings = model.encode([reconstructed_statement, abstracted_summary])
            similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

            return "correct" if similarity_score >= self.similarity_threshold else "incorrect"

    def extract_entity_from_question(self, question, linked_entities):
        doc = nlp(question)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        if entities:
            return entities[0][0]  # Return the first entity found

        # If no named entities are detected, attempt to match key tokens in the question
        tokens = [token.text for token in doc if token.pos_ in {"PROPN", "NOUN"}]
        for token in tokens:
            best_match = self.find_best_entity_match(token, linked_entities)
            if best_match:
                return best_match['name']

        raise ValueError("No entity found in the question.")

    def abstract_summary(self, summary):
        doc = nlp(summary)
        subj = [token.text for token in doc if token.dep_ == "nsubj"]
        root = [token.text for token in doc if token.dep_ == "ROOT"]
        obj = [token.text for token in doc if token.dep_ in {"attr", "dobj", "pobj"}]

        return " ".join([subj[0] if subj else "", root[0] if root else "", obj[0] if obj else ""]).strip()
