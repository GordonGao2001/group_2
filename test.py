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
from sklearn.metrics import accuracy_score
import json

nlp = spacy.load("en_core_web_sm")

bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
model_path = '/Users/gaoxiangyu/desktop/models/llama-2-7b.Q4_K_M.gguf'
# model_path = '/home/user/models/llama-2-7b.Q4_K_M.gguf' # in docker
# model_path = '/mnt/f/homeworks/WDPR/group_2/models/llama-2-7b-chat.Q4_K_M.gguf'
llm = Llama(model_path=model_path, verbose=False)

def calculate_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def load_test_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def evaluate_model(test_data, output_file_path, detailed_output_path, summary_output_path):
    detailed_results = []
    question_recognition_results = []
    answer_extraction_results = []
    fact_check_results = []

    for item in test_data:
        print(f"Processing Question ID: {item['id']}")
        question_id = item['id']
        question_text = item['question']
        expected_answer = item['expected_answer']
        question_type_test = item['type']

        # Generate raw answer using LLM
        output = llm(
            question_text,  # Prompt
            max_tokens=48,  # Generate up to 48 tokens
            echo=False  # Do not echo the prompt back in the output
        )
        raw_answer = output['choices'][0]['text'].strip()

        # Question type recognition
        my_q_type = question_type.question_type(question_text)
        question_type_correct = (my_q_type == 1 and question_type_test == "yes/no") or \
                                (my_q_type == 2 and question_type_test == "entity")
        question_recognition_results.append(question_type_correct)

        # Entity extraction and linking
        QandA = question_text + ' ' + raw_answer
        filtered_named_entities = extract_named_entities(QandA)
        if not filtered_named_entities:
            continue

        linked_entities = candidate_linking(question_text, raw_answer, filtered_named_entities, bert_model)
        if not linked_entities:
            continue

        if question_type_test == "yes/no":
            my_extract = sentence_sentiment.classify_yes_no(raw_answer)
            answer_correct = my_extract == expected_answer
            answer_extraction_results.append(answer_correct)

            fcr = FactChecker()
            result = fcr.fact_check(1, question_text, my_extract, [])
            fact_check_correct = (my_extract == expected_answer and result == "correct") or \
                                 (my_extract != expected_answer and result == "incorrect")
            fact_check_results.append(fact_check_correct)

        elif question_type_test == "entity":
            my_extract = ee.extract_entity_answer(linked_entities, raw_answer, bert_model)
            similarity = calculate_similarity(my_extract, expected_answer)
            answer_correct = similarity >= 0.5
            answer_extraction_results.append(answer_correct)

            matcher = Matcher()
            my_urls = [entity['url'] for entity in linked_entities]
            fcr = FactChecker()
            result = fcr.fact_check(2, question_text, my_extract, my_urls, extracted_entity=my_extract)
            fact_check_correct = (similarity >= 0.5 and result == "correct") or \
                                 (similarity < 0.5 and result == "incorrect")
            fact_check_results.append(fact_check_correct)

        else:
            continue

        # Record detailed results
        print(f"Detailed Result for Question ID {question_id}:")
        print(f"Question: {question_text}")
        if my_q_type==1:
            print(f"Classify Question type: yes/no, Test Question type: {question_type_test}")
        else:
            print(f"Classify Question type: entity, Test Question type: {question_type_test}")
        print(f"Question Type Correct: {question_type_correct}")
        
        print(f"Expected Answer: {expected_answer}")
        print(f"Extracted Answer: {my_extract}")

        print(f"Answer Correct: {answer_correct}")
        
        print(f"Fact Chect: {result}")
        print(f"Fact Check Correct: {fact_check_correct}")
        detailed_results.append({
            "id": question_id,
            "question": question_text,
            "question_type_correct": question_type_correct,
            "expected_answer": expected_answer,
            "my_answer": my_extract,
            "answer_correct": answer_correct,
            "fact_check_correct": fact_check_correct
        })

    # Write detailed results to file
    with open(detailed_output_path, "w") as detailed_file:
        json.dump(detailed_results, detailed_file, indent=4)

    # Calculate accuracies
    question_acc = sum(question_recognition_results) / len(question_recognition_results)
    answer_acc = sum(answer_extraction_results) / len(answer_extraction_results)
    fact_check_acc = sum(fact_check_results) / len(fact_check_results)

    # Write summary results to file
    summary_results = {
        "question_type_recognition_accuracy": question_acc,
        "answer_extraction_accuracy": answer_acc,
        "fact_checking_accuracy": fact_check_acc
    }
    with open(summary_output_path, "w") as summary_file:
        json.dump(summary_results, summary_file, indent=4)

    return summary_results

# Load test data
test_data = load_test_data('test.json')
# Evaluate model
results = evaluate_model(test_data, "output.txt", "test_output.json", "test_report.json")

# Print results
print(f"Question Type Recognition Accuracy: {results['question_type_recognition_accuracy']:.2%}")
print(f"Answer Extraction Accuracy: {results['answer_extraction_accuracy']:.2%}")
print(f"Fact Checking Accuracy: {results['fact_checking_accuracy']:.2%}")
