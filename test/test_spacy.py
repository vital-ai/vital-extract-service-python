import spacy

coref = spacy.load('en_coreference_web_trf')
nlp = spacy.load("en_core_web_trf")
# nlp.add_pipe("entityLinker", last=True)


def replace_pronouns(text, doc):
    replacements = []

    for cluster_key, mentions in doc.spans.items():
        main_mention = mentions[0]  # Assuming the first mention is the main mention
        for mention in mentions:
            # Ensure the mention is a pronoun and not the main mention
            if mention != main_mention and mention.text.lower() in ['he', 'she', 'it', 'they', 'him', 'her', 'them']:
                # Store replacement info (start position, end position, and replacement text)
                replacements.append((mention.start_char, mention.end_char, main_mention.text))

    # Sort replacements by start position in descending order to avoid messing up the indices
    replacements.sort(key=lambda x: x[0], reverse=True)

    # Create a new version of the text with replacements
    for start, end, replacement in replacements:
        text = text[:start] + replacement + text[end:]

    return text


def main():
    print('Hello World')

    text = """
    This is the first paragraph. It has two sentences.
    Here's the second paragraph with one sentence.

    John Smith was born in New York City on January 1, 1980.
    He married Jane Doe, who works for IBM, on March 13th, 2010.
    Also, he works for Google out of the offices in San Francisco.

    And here is another one with three sentences. This is the second sentence. Finally, the third sentence.
    """

    doc = coref(text)
    print(doc.spans)

    resolved_text = replace_pronouns(text, doc)

    paragraphs = [p for p in resolved_text.split('\n') if p]

    for i, paragraph in enumerate(paragraphs, start=1):
        print(f"Paragraph {i}:")
        doc = nlp(paragraph)
        for sent in doc.sents:
            print(f"  Sentence: {sent.text}")
            for ent in sent.ents:
                print(f"    Entity: {ent.text}, Type: {ent.label_}")
            # sent._.linkedEntities.pretty_print()

        # doc._.linkedEntities.print_super_entities()
        # for entity in doc._.linkedEntities:
        #    print(f"ID: {entity.get_id()}, LABEL: {entity.get_label()}, URL: {entity.get_url()}, Description: {entity.get_description()}")


if __name__ == "__main__":
    main()
