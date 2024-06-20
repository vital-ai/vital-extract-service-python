from typing import Dict

from transformers import AutoTokenizer, pipeline, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModelForQuestionAnswering
import torch
import spacy
# from transformers.models.t5.tokenization_t5_fast import T5Tokenizer
from transformers.models.t5.tokenization_t5_fast import T5Tokenizer

# extract service
# split sentences, paragraphs
# topic segmentation
# coreference resolution
# basic entity extraction
# relationship extraction
# entity to entity relations
# entity to property relations


class ExtractService:

    def __init__(self):
        self.coref = spacy.load('en_coreference_web_trf')
        self.nlp = spacy.load("en_core_web_trf")

        # self.summarizer = pipeline("summarization", model="t5-base")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base", model_max_length=1024)
        self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")

    def replace_pronouns(self, text, doc):

        replacements = []

        for cluster_key, mentions in doc.spans.items():
            main_mention = mentions[0]  # Assuming the first mention is the main mention
            for mention in mentions:
                # Ensure the mention is a pronoun and not the main mention
                if mention != main_mention and mention.text.lower() in ['he', 'she', 'it', 'they', 'him', 'her',
                                                                        'them']:
                    # Store replacement info (start position, end position, and replacement text)
                    replacements.append((mention.start_char, mention.end_char, main_mention.text))

        # Sort replacements by start position in descending order to avoid messing up the indices
        replacements.sort(key=lambda x: x[0], reverse=True)

        # Create a new version of the text with replacements
        for start, end, replacement in replacements:
            before = text[start:end]
            print(f"Replace {before} with {replacement}")
            text = text[:start] + replacement + text[end:]

        return text

    def extract(self, doc_id: str, doc_text: str) -> Dict:

        results = {}

        paragraph_list = []

        sentence_list = []

        entity_list = []

        entity_mention_map = {}

        summarize_input = "summarize: " + doc_text

        inputs = self.tokenizer(summarize_input, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")

        # print(f"inputs: {inputs}")

        # doc_summary = self.summarizer(summarize_input)

        output = self.t5_model.generate(**inputs, max_new_tokens=200)

        # print(f"output: {output}")

        doc_summary = self.tokenizer.decode(output[0], skip_special_tokens=True)

        print(f"summary: {doc_summary}")

        doc = self.coref(doc_text)

        # print(doc.spans)

        resolved_text = self.replace_pronouns(doc_text, doc)

        paragraphs = [p for p in resolved_text.split('\n') if p]

        for i, paragraph in enumerate(paragraphs, start=1):
            print(f"Paragraph {i}:")
            doc = self.nlp(paragraph)
            for sent in doc.sents:
                print(f"  Sentence: {sent.text}")
                for ent in sent.ents:
                    print(f"    Entity: {ent.text}, Type: {ent.label_}")

        return results
