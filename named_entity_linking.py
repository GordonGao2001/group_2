import wikipedia
import wikipediaapi
from wikipedia.exceptions import DisambiguationError, PageError
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
wiki_wiki = wikipediaapi.Wikipedia('WDPS Assignment Group_2', 'en')

def generate_candidates(entity_name):
    candidates_info = []
    try:
        search_results = wikipedia.search(entity_name, results=10)
        if not search_results:
            return []
        for candidate in search_results:
            try:
                candidate_page_api = wiki_wiki.page(candidate)
                if candidate_page_api.exists():
                    candidates_info.append({
                        'name': candidate_page_api.title,
                        'url': candidate_page_api.fullurl,
                        'summary': candidate_page_api.summary
                    })
                # Handle disambiguation by adding multiple entries
                elif isinstance(candidate_page_api,
                                wikipediaapi.WikipediaPage) and candidate_page_api.is_disambiguation():
                    for disambiguation_option in candidate_page_api.links.keys():
                        disambiguation_page = wiki_wiki.page(disambiguation_option)
                        if disambiguation_page.exists():
                            candidates_info.append({
                                'name': disambiguation_page.title,
                                'url': disambiguation_page.fullurl,
                                'summary': disambiguation_page.summary
                            })
            except Exception as e:
                print(f"Error fetching page for '{candidate}': {e}")
                continue

    except PageError:
        print(f"No page found for '{entity_name}'.")
    except Exception as e:
        print(f"Error during search for '{entity_name}': {e}")

    # Remove duplicates based on name and url
    unique_candidates = []
    seen = set()
    for candidate in candidates_info:
        identifier = (candidate['name'], candidate['url'])
        if identifier not in seen:
            seen.add(identifier)
            unique_candidates.append(candidate)
    return unique_candidates



def candidate_linking(question, answer, named_entities, bert_model):
    linked_entities = []
    seen_entities = set()
    normalized_entities = set()
    
    # Encode context (question and answer) into vectors
    question_vector = bert_model.encode([question])[0]
    answer_vector = bert_model.encode([answer])[0]
    context_vector = question_vector * 0.5 + answer_vector * 0.5

    for named_entity, label in named_entities:
        normalized_text = named_entity.lower()
        if normalized_text in normalized_entities:
            continue
        normalized_entities.add(normalized_text)

        # Generate candidate entities
        candidate_entities_info = generate_candidates(named_entity)

        if candidate_entities_info:
            candidate_texts = [info['summary'] for info in candidate_entities_info]
            candidate_links = [info['url'] for info in candidate_entities_info]
            candidate_vectors = [bert_model.encode([text])[0] for text in candidate_texts]

            # Compute cosine similarity
            similarities = cosine_similarity([context_vector], candidate_vectors)[0]
            weighted_similarities = []

            for info, similarity in zip(candidate_entities_info, similarities):
                # Compute name similarity
                name_similarity = SequenceMatcher(None, info['name'].lower(), named_entity.lower()).ratio()
                if info['name'].lower() == named_entity.lower():
                    name_similarity = 1.0
                elif named_entity.lower() in info['name'].lower():
                    name_similarity += 0.2

                # Combine similarities
                weighted_similarity = similarity * 0.5 + name_similarity * 0.5
                weighted_similarities.append((info, weighted_similarity))

            # Rank candidates
            ranked_candidates = sorted(weighted_similarities, key=lambda x: x[1], reverse=True)

            # Select top candidate
            if ranked_candidates:
                top_candidate = ranked_candidates[0][0]
                identifier = (top_candidate['name'], top_candidate['url'])
                if identifier not in seen_entities:
                    seen_entities.add(identifier)
                    linked_entities.append({
                        'name': top_candidate['name'],
                        'url': top_candidate['url'],
                        'similarity': ranked_candidates[0][1],
                        'summary': top_candidate['summary']  # Add the summary attribute
                    })

    return linked_entities
