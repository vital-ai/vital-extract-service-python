import re
import spacy
from spacy import Language
from spacy.tokens import DocBin, Span, Doc
import os
from transformers import AutoModel, AutoTokenizer
import dill
from extractservice.spacy.spacy_pipeline import SpacyPipeline
from extractservice.spacy.utils.spacy_utils import SpacyUtils


def get_data_file_content(file_path: str) -> str:
    with open(file_path, 'r') as file:
        file_contents = file.read()

    return file_contents


def split_into_chapters(text):
    # This pattern captures 'CHAPTER' followed by any characters that do not start a new 'CHAPTER' line
    pattern = r'(CHAPTER [^\n]*\n(?:.*?)(?=\nCHAPTER |\Z))'
    # Using re.DOTALL to make . match any character including newline
    chapters = re.findall(pattern, text, re.DOTALL)

    return chapters


def merge_docs(parent_doc, sub_docs, nlp):

    parent_doc.spans["paragraphs"] = []
    parent_doc.spans["sentences"] = []
    parent_doc.spans["entities"] = []
    parent_doc.spans["coref_clusters"] = []
    parent_doc.spans["corefs"] = []

    offset = 0

    paragraph_count = 1

    for doc in sub_docs:

        paragraph_span = parent_doc.char_span(offset, offset + len(doc.text), label="paragraph")

        if paragraph_span:
            paragraph_span._.is_paragraph = True
            parent_doc.spans["paragraphs"].append(paragraph_span)

        for sent in doc.sents:
            start = offset + sent.start_char
            end = offset + sent.end_char
            span = parent_doc.char_span(start, end, label="sentence")
            if span:
                parent_doc.spans["sentences"].append(span)

        for ent in doc.ents:
            start = offset + ent.start_char
            end = offset + ent.end_char
            span = parent_doc.char_span(start, end, label=ent.label_)
            if span:
                parent_doc.spans["entities"].append(span)

        for coref_cluster in doc._.coref_clusters:

            cluster_start = coref_cluster.start
            cluster_end = coref_cluster.end
            cluster_text = coref_cluster.text

            paragraph_cluster_id = coref_cluster._.coref_id

            cluster_id = f"{paragraph_count}.{paragraph_cluster_id}"

            print(f"CoRef Cluster ID: {cluster_id}")

            print(f"CoRef Cluster Text: {cluster_text}")

            cluster_start_offset = offset + coref_cluster.start_char
            cluster_end_offset = offset + coref_cluster.end_char

            coref_cluster_span = parent_doc.char_span(
                cluster_start_offset,
                cluster_end_offset,
                label="COREF_CLUSTER")

            if coref_cluster_span is None:
                print(f"None span: start {cluster_start_offset}")
                print(f"None span: end {cluster_end_offset}")
                continue

            coref_cluster_span._.coref_id = cluster_id

            if coref_cluster_span:
                print(f"Appending CoRef Cluster Text: {cluster_text}")
                parent_doc.spans["coref_clusters"].append(coref_cluster_span)

            mentions = [mention for mention in doc.spans["coref_mentions"] if
                        mention.start >= cluster_start and mention.end <= cluster_end]

            for mention in mentions:

                try:
                    mention_start = mention.start
                    mention_start = mention.start
                    mention_end = mention.end
                    mention_text = mention.text

                    # this should be tracking the token pointed to
                    # not the literal text
                    mention_root = mention._.root_text

                    num_tokens = len(mention_text.split())

                    end = mention_start + num_tokens

                    start_offset = offset + doc[mention_start].idx

                    end_offset = offset + doc[mention_end].idx

                    parent_text = parent_doc.text[start_offset:end_offset]

                    print(f"CoRef Text: {mention_text} Parent Text: {parent_text}")

                    span = parent_doc.char_span(start_offset, end_offset, label="COREF")

                    if span:
                        span._.root_text = mention_root
                        span._.full_text = mention_text

                        span._.coref_id = cluster_id

                        parent_doc.spans["corefs"].append(span)

                except:
                    continue

        paragraph_count += 1

        offset += len(doc.text) + 2  # Adjusting offset for paragraphs separated by '\n\n'

    return parent_doc


def process_document(doc_text, nlp):

    paragraphs = doc_text.split('\n\n')  # Assuming paragraphs are separated by double newlines

    sub_docs = [nlp(paragraph) for paragraph in paragraphs if paragraph.strip()]

    parent_doc = nlp.make_doc(doc_text)

    parent_doc = merge_docs(parent_doc, sub_docs, nlp)

    return parent_doc


def main():
    print('Named Entity Extract')


    # read source text file

    file_path = '../test_data_internal/JK_Rowling_HarryPotter1_Sorcerers Stone.txt'

    file_contents = get_data_file_content(file_path)

    # split into sections
    chapters = split_into_chapters(file_contents)

    for i, chapter in enumerate(chapters, 1):
        pass
        # print(f"Chapter {i}:\n{chapter}\n{'-' * 20}")

    print(f'Total chapters: {len(chapters)}')

    # create doc for each section

    # process section with nlp pipeline
    # split into paragraphs
    # split into sentences
    # name entity mentions
    # co-reference

    # doc = nlp("'I will go to the store' said John.")
    # doc = nlp("Philip plays the bass because he loves it.")

    nlp = SpacyPipeline.setup_spacy()

    # doc_text = chapters[0]
    # doc = process_document(doc_text, nlp)
    # doc_list = [doc]
    # SpacyUtils.display_doc(doc)

    # with open('../test_output/hp_docs.1.dill', 'wb') as f:
    #    dill.dump(doc_list, f)

    chapter_count = 1
    
    doc_list = []
    
    for doc_text in chapters:

        print(f"Processing Chapter: {chapter_count}")

        doc = process_document(doc_text, nlp)
        
        doc_list.append(doc)

        chapter_count += 1

    with open('../test_output/hp_docs.2.dill', 'wb') as f:
        dill.dump(doc_list, f)


if __name__ == "__main__":
    main()
