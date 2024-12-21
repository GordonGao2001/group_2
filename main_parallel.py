import os
import concurrent.futures
import wikipedia
import wikipediaapi
from wikipedia.exceptions import DisambiguationError, PageError
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
import sentence_sentiment, question_type
import Entity_extr as ee
from matcher import Matcher
from fact_check_reconstruct import FactChecker
from named_entity_extraction import extract_named_entities
from named_entity_linking import candidate_linking

nlp = spacy.load("en_core_web_sm")
bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
model_path = "models/llama-2-7b-chat.Q4_K_M.gguf"
llm = Llama(model_path=model_path, verbose=False)

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
            except ValueError as e:
                print(f"Wrong file format in line: {line}. Error: {e} Exiting...")
                exit()
    return questions

def process_question(question_id, question_text):
    output = llm(question_text, max_tokens=48, echo=False)
    raw_answer = output['choices'][0]['text'].strip()
    R = question_id + '\t' + 'R' + '"' + raw_answer + '"' + '\n'
    
    QandA = question_text + ' ' + raw_answer
    filtered_named_entities = extract_named_entities(QandA)
    linked_entities = candidate_linking(question_text, raw_answer, filtered_named_entities, bert_model)
    
    my_q_type = question_type.question_type(question_text)
    response_data = [(question_id, R)]
    if not linked_entities:
        return response_data

    if my_q_type == 1:
        my_extract = sentence_sentiment.classify_yes_no(raw_answer)
        A = question_id + '\t' + 'A' + '\"' + my_extract + '\"\n'
        response_data.append((question_id, A))
    elif my_q_type == 2:
        my_extract = ee.extract_entity_answer(linked_entities, raw_answer, bert_model)
        matcher = Matcher()
        my_url = matcher.url(my_extract, linked_entities)
        A = question_id + '\t' + 'A' + '\"' + my_extract + '\"' + '\t' + '\"' + my_url + '\"\n'
        response_data.append((question_id, A))

    fcr = FactChecker()
    my_urls = [entity['url'] for entity in linked_entities]
    if my_q_type == 1:
        result = fcr.fact_check(my_q_type, question_text, my_extract, my_urls)
    else:
        result = fcr.fact_check(my_q_type, question_text, my_extract, my_urls, extracted_entity=my_extract)
    C = question_id + '\t' + 'C' + '\"' + result + '\"\n'
    response_data.append((question_id, C))

    for linked_entity in linked_entities:
        E = question_id + '\t' + 'E' + '"' + linked_entity['name'] + '"' + '\t' + '"' + linked_entity['url'] + '"' + '\n'
        response_data.append((question_id, E))

    return response_data

def main():
    q_list = file_reader(os.getcwd() + '/input.txt')
    results = [None] * len(q_list)  # Prepare results list of correct size

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        future_to_index = {executor.submit(process_question, q_id, q_text): i for i, (q_id, q_text) in enumerate(q_list)}
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()  # Store results at the correct index

    output_file_path = "output.txt"
    with open(output_file_path, "w") as output_file:
        for result in results:
            for item in result:
                output_file.write(item[1])  # Write the second item which contains the formatted string

if __name__ == "__main__":
    main()