import wikipedia
import wikipediaapi
from wikipedia.exceptions import DisambiguationError, PageError
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import spacy
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
import sentence_sentiment, question_type
import Entity_extr as ee
from matcher import Matcher
from fact_check_reconstruct import FactChecker

wiki_wiki = wikipediaapi.Wikipedia('WDPS Assignment Group_2', 'en')
nlp = spacy.load("en_core_web_sm")
bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
model_path = '/mnt/f/homeworks/WDPR/group_2/models/llama-2-7b-chat.Q4_K_M.gguf'
llm = Llama(model_path=model_path, verbose=False)


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

    question_vector = bert_model.encode([question])[0]
    answer_vector = bert_model.encode([answer])[0]

    question_weight = 0.5
    answer_weight = 0.5
    context_vector = question_vector * question_weight + answer_vector * answer_weight

    for named_entity, label in named_entities:
        normalized_text = named_entity.lower()
        if normalized_text in normalized_entities:
            continue
        seen_entities.add(named_entity)
        normalized_entities.add(normalized_text)
        candidate_entities_info = generate_candidates(named_entity)

        if candidate_entities_info:
            candidate_texts = [candidate_info['summary'] for candidate_info in candidate_entities_info]
            candidate_links = [candidate_info['url'] for candidate_info in candidate_entities_info]
            candidate_vectors = [bert_model.encode([text])[0] for text in candidate_texts]

            # Compute cosine similarity between context and candidate vectors
            similarities = cosine_similarity([context_vector], candidate_vectors)[0]
            weighted_similarities = []
            for candidate_info, similarity in zip(candidate_entities_info, similarities):
                name_similarity = SequenceMatcher(None, candidate_info['name'].lower(), named_entity.lower()).ratio()
                if candidate_info['name'].lower() == named_entity.lower():
                    name_similarity = 1.0
                elif named_entity.lower() in candidate_info['name'].lower():
                    name_similarity += 0.2

                # Adjust weights
                weighted_similarity = similarity * 0.5 + name_similarity * 0.5
                weighted_similarities.append((candidate_info, weighted_similarity))

            # Rank candidates
            ranked_candidates = sorted(weighted_similarities, key=lambda x: x[1], reverse=True)

            # Select the top-ranked candidate
            if ranked_candidates:
                top_ranked_candidate_info = ranked_candidates[0][0]
                top_ranked_candidate = {
                    'name': top_ranked_candidate_info['name'],
                    'url': top_ranked_candidate_info['url'],
                    'similarity': ranked_candidates[0][1]
                }

                # Ensure uniqueness of linked entities based on name and url
                identifier = (top_ranked_candidate['name'], top_ranked_candidate['url'])
                if identifier not in seen_entities:
                    seen_entities.add(identifier)
                    linked_entities.append(top_ranked_candidate)
    return linked_entities


# Read questions from input.txt, return (question_id, question_text) list
def file_reader(f_path):
    questions = []
    with open(f_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                match = re.match(r'^(\S+)\s+(.*)$', line)
                if match:
                    question_id, question_text = match.groups()
                    questions.append((question_id, question_text))
                else:
                    raise ValueError('Invalid format in current line!')
            except ValueError:
                print(f"Wrong file format in line: {line} Exiting...")
                exit()
    return questions


# Read questions from input file
q_list = []
input_file = os.getcwd() + '/input.txt'
if os.path.exists(input_file):
    q_list = file_reader(input_file)
else:
    print(f'input file not found\n should be at {input_file}\n, exiting')
    exit()

q_types = question_type.questions_classifier(q_list)
# print(q_types)
# Create output.txt
output_file_path = "output.txt"
output_file = open(output_file_path, "w")

for question_id, question_text in q_list:
    output = llm(
        question_text,  # Prompt
        max_tokens=48,  # Generate up to 48 tokens
        echo=False  # Do not echo the prompt back in the output
    )

    raw_answer = output['choices'][0]['text'].strip()
    R = question_id + '\t' + 'R' + '"' + raw_answer + '"' + '\n'
    output_file.write(R)

    print(question_text)
    print(R)

    # Extract entities from both question_text and raw_answer
    QandA = question_text + ' ' + raw_answer
    doc = nlp(QandA)

    # Extract named entities and remove duplicates
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]
    filtered_named_entities = []
    seen_entities = set()
    normalized_entities = set()
    for ent in doc.ents:
        normalized_text = ent.text.lower()
        if normalized_text in normalized_entities:
            continue
        seen_entities.add(ent.text)
        normalized_entities.add(normalized_text)
        # If an entity has more than one word like "King of England", keep it
        if len(ent) > 1:
            filtered_named_entities.append((ent.text, ent.label_))
        else:
            # If an entity is single word, check it
            if ent.root.pos_ in {"PROPN", "NOUN"}:
                filtered_named_entities.append((ent.text, ent.label_))

    print(filtered_named_entities)

    # Link entities
    linked_entities = candidate_linking(question_text, raw_answer, filtered_named_entities, bert_model)

    # extract entities or yes/no
    my_q_type = q_types[int(question_id[-3:]) - 1]
    if my_q_type == 1:  # yes/no case
        my_extract = sentence_sentiment.classify_yes_no(raw_answer)
        A = question_id + '\t' + 'A' + '\"' + my_extract + '\"\n'
        print(A)
        output_file.write(A)
    elif my_q_type == 2:  # entity case
        my_extract = ee.extract_entity_answer(linked_entities, raw_answer, bert_model)
        matcher = Matcher()
        my_url = matcher.url(my_extract, linked_entities)
        A = question_id + '\t' + 'A' + '\"' + my_extract + '\"'+ '\t'+ '\"'+ my_url + '\"\n'
        print(A)
        output_file.write(A)
    else:
        continue


    # Fact checking
    fcr = FactChecker()
    if my_q_type == 1:
        result = fcr.fact_check(my_q_type, question_text, raw_answer, my_q_type, my_url, extracted_entity=my_extract)
        C = question_id + '\t' + 'C' + '\"' + result + '\"\n'
        print(C)
        output_file.write(C)
    else:
        result = fcr.fact_check(my_q_type, question_text, raw_answer, my_q_type, my_url, extracted_entity=my_extract)
        C = question_id + '\t' + 'C' + '\"' + result + '\"\n'
        print(C)
        output_file.write(C)

    for linked_entity in linked_entities:
        E = question_id + '\t' + 'E' + '"' + linked_entity['name'] + '"' + '\t' + '"' + linked_entity[
            'url'] + '"' + '\n'
        print(E)
        output_file.write(E)
        
output_file.close()
