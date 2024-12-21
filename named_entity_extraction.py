import spacy
nlp = spacy.load("en_core_web_sm")

def extract_named_entities(text):
    doc = nlp(text)
    valid_entity_labels = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "FAC", "WORK_OF_ART"}
    filtered_named_entities = []
    seen_entities = set()

    for ent in doc.ents:
        normalized_text = ent.text.lower().strip()
        if normalized_text in seen_entities:
            continue
        seen_entities.add(normalized_text)
        if ent.label_ in valid_entity_labels:
            filtered_named_entities.append((ent.text.strip(), ent.label_))

    tokens = [token for token in doc if token.pos_ in {"PROPN", "NOUN"}]
    temp_entity = []
    for token in tokens:
        if token.ent_iob_ == "O" or token.pos_ in {"PROPN", "NOUN"}:
            temp_entity.append(token.text)
        else:
            if temp_entity:
                potential_entity = " ".join(temp_entity).strip()
                if len(potential_entity.split()) <= 5 and potential_entity.lower() not in seen_entities:
                    seen_entities.add(potential_entity.lower())
                    filtered_named_entities.append((potential_entity, "UNKNOWN"))
                temp_entity = []

    if temp_entity:
        potential_entity = " ".join(temp_entity).strip()
        if len(potential_entity.split()) <= 5 and potential_entity.lower() not in seen_entities:
            seen_entities.add(potential_entity.lower())
            filtered_named_entities.append((potential_entity, "UNKNOWN"))

    for i, token in enumerate(doc):
        if token.text.lower() == "of" and i > 0 and i < len(doc) - 1:
            left_token = doc[i - 1]
            right_token = doc[i + 1]
            if left_token.pos_ in {"PROPN", "NOUN"} and right_token.pos_ in {"PROPN", "NOUN"}:
                potential_entity = f"{left_token.text} of {right_token.text}".strip()
                if potential_entity.lower() not in seen_entities:
                    seen_entities.add(potential_entity.lower())
                    filtered_named_entities.append((potential_entity, "UNKNOWN"))

    if not filtered_named_entities:
        for token in doc:
            if token.pos_ == "NOUN" and token.text.lower() not in seen_entities:
                filtered_named_entities.append((token.text, "UNKNOWN"))
                break

    return filtered_named_entities
