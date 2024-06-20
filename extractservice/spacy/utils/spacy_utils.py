from spacy import Language
import re
from spacy.tokens import DocBin, Span, Doc
import spacy

class SpacyUtils:

    @staticmethod
    def display_doc(doc):

        for paragraph in doc.spans["paragraphs"]:

            # print(f"Paragraph: {paragraph.text}")

            # print(f"Paragraph Start: {paragraph.start}")
            # print(f"Paragraph End: {paragraph.end}")

            sentences = [sent for sent in doc.spans["sentences"] if
                         sent.start >= paragraph.start and sent.end <= paragraph.end]

            entities = [ent for ent in doc.spans["entities"] if
                        ent.start >= paragraph.start and ent.end <= paragraph.end]

            # corefs = [coref for coref in doc.spans["corefs"] if
            #          coref.start >= paragraph.start and coref.start <= paragraph.end]

            for sent in sentences:
                print(f"  Sentence: {sent.text}")

            for ent in entities:
                print(f"  Entity: {ent.text} ({ent.label_})")

            # for coref in corefs:
            #    print(f"  Coreference Entity: {coref.text}")
            # print(f"  Mentions: {', '.join(coref['data']['mentions'])}")

            print("\n")

        coref_clusters = doc.spans["coref_clusters"]

        corefs = doc.spans["corefs"]

        print(f"Coref Cluster Count: {len(coref_clusters)}")
        print(f"Coref Count: {len(corefs)}")

        for coref_cluster in doc.spans["coref_clusters"]:

            # print(f"  Coreference Cluster Entity: {coref_cluster.text}")

            cluster_id = coref_cluster._.coref_id

            # print(f"  Coreference Cluster ID: {cluster_id}")

            for mention_span in doc.spans["corefs"]:

                mention_id = mention_span._.coref_id

                if mention_id == cluster_id:
                    # print(f"    Contained Mention ID: {mention_id}")
                    print(f"    Contained Mention: {mention_span._.full_text}")
                    # print(f"    Contained Mention Root: {mention_span._.root_text}")

                if coref_cluster.start <= mention_span.start and coref_cluster.end >= mention_span.end:
                    pass

                    # there may be off by filtering going on?
                    # mention_id = mention_span._.coref_id
                    # print(f"    Contained Mention ID: {mention_id}")
                    # print(f"    Contained Mention: {mention_span.text}")


