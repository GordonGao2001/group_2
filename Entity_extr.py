from sklearn.metrics.pairwise import cosine_similarity
def extract_entity_answer(named_entities, raw_answer, bert_model):
    # Encode the raw answer
    answer_vector = bert_model.encode([raw_answer])[0]

    # Score entities based on semantic similarity
    best_entity = None
    best_score = 0
    for entity in named_entities:
        entity_name = entity['name']
        entity_vector = bert_model.encode([entity_name])[0]
        similarity_score = cosine_similarity([answer_vector], [entity_vector])[0][0]
        if similarity_score > best_score:
            best_score = similarity_score
            best_entity = entity

    return best_entity['name'] if best_entity else "#"