import dill
import spacy
import json
from extractservice.spacy.spacy_pipeline import SpacyPipeline


def span_to_dict(span):
    span_dict = {
        "text": span.text,
        "start": span.start_char,
        "end": span.end_char
    }
    if span.label_:
        span_dict["label"] = span.label_
    if span.has_extension("is_paragraph") and span._.is_paragraph is not None:
        span_dict["is_paragraph"] = span._.is_paragraph
    return span_dict


def doc_to_dict(doc):
    paragraphs = []
    for paragraph in doc.spans["paragraphs"]:
        sentences = [span_to_dict(sent) for sent in doc.spans["sentences"] if sent.start >= paragraph.start and sent.end <= paragraph.end]
        entities = [span_to_dict(ent) for ent in doc.spans["entities"] if ent.start >= paragraph.start and ent.end <= paragraph.end]
        paragraph_dict = span_to_dict(paragraph)
        if sentences:
            paragraph_dict["sentences"] = sentences
        if entities:
            paragraph_dict["entities"] = entities
        paragraphs.append(paragraph_dict)

    doc_dict = {
        "text": doc.text,
        "paragraphs": paragraphs
    }
    return doc_dict


def main():

    print('Test Doc Export')

    nlp = SpacyPipeline.setup_spacy()

    with open('../test_output/hp_docs.2.dill', 'rb') as f:
        doc_list = dill.load(f)

    data = []

    with open("../test_output/hp_docs.2.jsonnl", "w", encoding="utf-8") as f:
        for doc in doc_list:
            doc_dict = doc_to_dict(doc)
            f.write(json.dumps(doc_dict) + "\n")


if __name__ == "__main__":
    main()

