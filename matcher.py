from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Matcher:
    def __init__(self):
        # Initialize the vectorizer for entity name comparisons
        self.vectorizer = TfidfVectorizer()

    def url(self, extracted_entity, linked_entities):
        """
        Match the extracted entity to the most similar entity in the linked_entities list
        and return its URL.

        Parameters:
            extracted_entity (str): The name of the extracted entity.
            linked_entities (list of dict): List of entities with 'name' and 'url' keys.

        Returns:
            str: The URL of the best-matching entity.
        """
        # Preprocess entity names from linked_entities
        entity_names = [entity['name'] for entity in linked_entities]

        # Combine extracted entity and entity names into a single list for vectorization
        all_entities = [extracted_entity] + entity_names
        tfidf_matrix = self.vectorizer.fit_transform(all_entities)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        best_match_index = similarities.argmax()
        # print(f'best match of entity {extracted_entity} has index {best_match_index}')
        return linked_entities[best_match_index]['url']

def test():
    # Example data
    linked_entities = [
        {"name": "China", "url": "https://en.wikipedia.org/wiki/China"},
        {"name": "Chinese People", "url": "https://en.wikipedia.org/wiki/Chinese_people"},
        {"name": "Republic of China", "url": "https://en.wikipedia.org/wiki/Republic_of_China"}
    ]

    extracted_entity = "Chinas"  # Slight typo in the name

    # Initialize Matcher and find the URL
    matcher = Matcher()
    matched_url = matcher.url(extracted_entity, linked_entities)

    print(f"Extracted Entity: {extracted_entity}")
    print(f"Matched URL: {matched_url}")

# test()