from spacy import Language
import re
from spacy.tokens import DocBin, Span, Doc
import spacy


class SpacyPipeline:

    @staticmethod
    @Language.component("custom_coref_component")
    def custom_coref_component(doc):

        doc.spans["coref_clusters"] = []
        doc.spans["coref_mentions"] = []

        # Iterate over each cluster by examining the keys that start with 'coref_clusters'
        for key in doc.spans:
            if key.startswith('coref_clusters_'):

                cluster_id = None

                match = re.search(r'\d+$', key)

                if match:
                    cluster_id = int(match.group())

                spans = doc.spans[key]

                # print(f"processing cluster: {key} count: {len(spans)}")

                if spans:
                    # Calculate the span that encompasses the entire cluster
                    cluster_start = min(span.start for span in spans)
                    cluster_end = max(span.end for span in spans)

                    # Create a new span that covers the whole cluster
                    cluster_span = Span(doc, cluster_start, cluster_end, label="COREF_CLUSTER")

                    cluster_span._.coref_id = cluster_id

                    doc.spans["coref_clusters"].append(cluster_span)

                    # Optionally store the detailed information for each mention within the cluster
                    # coref_cluster_spans.append([(span.root.text, span.text, span.start) for span in spans])

                    # TODO need to keep the ID so the resolved entities
                    # can be grouped together

                    for span in spans:
                        mention_span = Span(doc, span.start, span.end, label="COREF")

                        mention_span._.coref_id = cluster_id

                        # we want to track the token here not the literal text
                        mention_span._.root_text = span.root.text

                        doc.spans["coref_mentions"].append(mention_span)

        doc._.coref_clusters = doc.spans["coref_clusters"]
        doc._.coref_mentions = doc.spans["coref_mentions"]

        return doc

    @staticmethod
    def setup_spacy():

        # os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # os.environ["HF_DATASETS_OFFLINE"] = "1"
        # os.environ["HF_METRICS_OFFLINE"] = "1"

        nlp = spacy.load('en_coreference_web_trf')

        # if "experimental_coref" not in nlp.pipe_names:
        #    nlp.add_pipe("experimental_coref")

        core = spacy.load("en_core_web_trf")

        nlp.replace_listeners("transformer", "coref", ["model.tok2vec"])
        nlp.replace_listeners("transformer", "span_resolver", ["model.tok2vec"])

        nlp.remove_pipe("transformer")

        for pipe in core.pipe_names:
            nlp.add_pipe(pipe, source=core)

        if not Span.has_extension("is_paragraph"):
            Span.set_extension("is_paragraph", default=False)

        if not Doc.has_extension("coref_clusters"):
            Doc.set_extension("coref_clusters", default=None)

        if not Doc.has_extension("coref_mentions"):
            Doc.set_extension("coref_mentions", default=None)

        if not Span.has_extension("root_text"):
            Span.set_extension("root_text", default=None)

        if not Span.has_extension("full_text"):
            Span.set_extension("full_text", default=None)

        if not Span.has_extension("coref_id"):
            Span.set_extension("coref_id", default=None)

        # print(nlp.pipe_names)

        nlp.add_pipe("custom_coref_component", last=True)

        # nlp.initialize()

        return nlp

