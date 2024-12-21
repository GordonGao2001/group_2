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
from named_entity_extraction import extract_named_entities
from named_entity_linking import candidate_linking
#from fact_checking_with_GPT import validate_extracted_answer
nlp = spacy.load("en_core_web_sm")

bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# model_path = '/Users/gaoxiangyu/desktop/models/llama-2-7b.Q4_K_M.gguf'
model_path = '/home/user/models/llama-2-7b.Q4_K_M.gguf' # in docker
# model_path = '/mnt/f/homeworks/WDPR/group_2/models/llama-2-7b-chat.Q4_K_M.gguf'
llm = Llama(model_path=model_path, verbose=False)


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

    # Extract named entities and remove duplicates
    filtered_named_entities = extract_named_entities(QandA)
    if not filtered_named_entities:
        continue
    # print(filtered_named_entities)
    # Link entities
    linked_entities = candidate_linking(question_text, raw_answer, filtered_named_entities, bert_model)
    if not linked_entities:
        continue
    

    # my_q_type = q_types[int(question_id[-3:]) - 1]
    # my_q_type = question_type.questions_classifier([question_text])[0]
    my_q_type = question_type.question_type(question_text)
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
    my_urls = [entity['url'] for entity in linked_entities]
    if my_q_type == 1:
        result = fcr.fact_check(my_q_type, question_text, my_extract, my_urls)
        C = question_id + '\t' + 'C' + '\"' + result + '\"\n'
        print(C)
        output_file.write(C)
    else:
        result = fcr.fact_check(my_q_type, question_text, my_extract, my_urls, extracted_entity=my_extract)
        C = question_id + '\t' + 'C' + '\"' + result + '\"\n'
        print(C)
        output_file.write(C)


    for linked_entity in linked_entities:
        E = question_id + '\t' + 'E' + '"' + linked_entity['name'] + '"' + '\t' + '"' + linked_entity[
            'url'] + '"' + '\n'
        print(E)
        print(linked_entity)
        output_file.write(E)
    


output_file.close()
