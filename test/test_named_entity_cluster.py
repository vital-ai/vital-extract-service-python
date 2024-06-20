import dill
from spacy import Language
import re
from spacy.tokens import DocBin, Span, Doc
import spacy
from extractservice.spacy.spacy_pipeline import SpacyPipeline
from extractservice.spacy.utils.spacy_utils import SpacyUtils
import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan
import pandas


def find_unique_names_by_cluster_id(data, tree, cluster_id):

    unique_names = set()

    tree_df = tree.to_pandas()

    cluster_tree = tree_df[tree_df.child_size > 1]

    cluster_tree = cluster_tree.sort_values(by=['parent', 'lambda_val'], ascending=[True, False])

    roots = cluster_tree[~cluster_tree.parent.isin(cluster_tree.child)]

    for _, row in roots.iterrows():

        cid = int(row.child)

        def walk_node(node_id, match: bool):

            # print(f"Walk: {node_id} Match: {match}")

            node_names = []

            if node_id == cluster_id:
                match = True

            for d in data:
                # print(d['cluster_id'])
                if d['cluster_id'] == node_id:
                    print(d)
                    if match:
                        node_names.append(d['name'])

            children = cluster_tree[cluster_tree.parent == node_id]

            if not children.empty:
                for _, row in children.iterrows():
                    child_id = int(row.child)
                    child_names = walk_node(child_id, match)
                    for c in child_names:
                        node_names.append(c)

            return node_names

        if cluster_id == cid:
            root_names = walk_node(cid, True)
        else:
            root_names = walk_node(cid, False)

        for r in root_names:
            unique_names.add(r)

    return unique_names


def find_final_clusters_in_top_cluster(data, tree, top_cluster_id, clusters):
    tree_df = tree.to_pandas()

    # Function to recursively collect all points under a given cluster node
    def collect_points(cluster_id):
        points = set()
        children = tree_df[tree_df['parent'] == cluster_id]
        for _, child in children.iterrows():
            if child['child_size'] == 1:  # It's a leaf node
                points.add(int(child['child']))
            else:
                points.update(collect_points(int(child['child'])))
        return points

    # Collect all points starting from the top_cluster_id
    points_in_top_cluster = collect_points(top_cluster_id)

    # Map these points to their final cluster IDs
    final_cluster_mapping = {point: clusters[point] for point in points_in_top_cluster if clusters[point] != -1}

    # Retrieve the corresponding data items, mapping them to their final clusters
    final_clusters_data = {cluster_id: [] for cluster_id in set(final_cluster_mapping.values())}
    for point, cluster_id in final_cluster_mapping.items():
        final_clusters_data[cluster_id].append(data[point])

    return final_clusters_data


def print_cluster_hierarchy(data, tree, clusters):
    # Convert tree to a pandas DataFrame
    tree_df = tree.to_pandas()
    # Select only the clusters (exclude single points)
    cluster_tree = tree_df[tree_df.child_size > 1]
    # Sort by the parent then by lambda value in descending order
    cluster_tree = cluster_tree.sort_values(by=['parent', 'lambda_val'], ascending=[True, False])

    def print_node(cluster_id, indent=0):
        # Find children of the current cluster
        children = cluster_tree[cluster_tree.parent == cluster_id]
        if not children.empty:
            for _, row in children.iterrows():

                print(' ' * indent + f'Cluster {int(row.child)}, Size: {int(row.child_size)}')

                internal_cluster_id = int(row.child)

                clusters_data = find_final_clusters_in_top_cluster(data, tree, internal_cluster_id, clusters)

                unique_names = set()

                for cluster_id, items in clusters_data.items():
                    # print(f"Internal Cluster ID: {internal_cluster_id} Cluster ID {cluster_id} contains: {items}")
                    for item in items:
                        name = item['name']
                        unique_names.add(name)

                print(', '.join(unique_names))

                # entity_names = find_unique_names_by_cluster_id(data, tree, int(row.child))

                # print(', '.join(entity_names))

                # Recurse to print each child
                print_node(row.child, indent + 4)

    # Find the roots
    roots = cluster_tree[~cluster_tree.parent.isin(cluster_tree.child)]
    for _, row in roots.iterrows():
        print(f'Cluster {int(row.child)}, Size: {int(row.child_size)}')

        # entity_names = find_unique_names_by_cluster_id(data, tree, int(row.child))

        # print(', '.join(entity_names))

        print_node(row.child, 4)


def filter_cluster_hierarchy(data, tree, clusters):

    filtered_nodes = []

    tree_df = tree.to_pandas()
    cluster_tree = tree_df[tree_df.child_size > 1]
    cluster_tree = cluster_tree.sort_values(by=['parent', 'lambda_val'], ascending=[True, False])

    def filter_node(cluster_id):

        children = cluster_tree[cluster_tree.parent == cluster_id]

        if not children.empty:
            for _, row in children.iterrows():

                # print(f'Cluster {int(row.child)}, Size: {int(row.child_size)}')

                internal_cluster_id = int(row.child)

                clusters_data = find_final_clusters_in_top_cluster(data, tree, internal_cluster_id, clusters)

                unique_names = set()

                for cluster_id, items in clusters_data.items():
                    # print(f"Internal Cluster ID: {internal_cluster_id} Cluster ID {cluster_id} contains: {items}")
                    for item in items:
                        name = item['name']
                        unique_names.add(name)

                if len(unique_names) <= 4:
                    filter_map = {"names": unique_names, "internal_cluster_id": internal_cluster_id, "cluster_id": cluster_id}
                    filtered_nodes.append(filter_map)
                else:
                    filter_node(row.child)

    roots = cluster_tree[~cluster_tree.parent.isin(cluster_tree.child)]

    for _, row in roots.iterrows():

        filter_node(row.child)

    return filtered_nodes


def filter_cluster_hierarchy(data, tree, clusters):

    filtered_nodes = []

    tree_df = tree.to_pandas()
    cluster_tree = tree_df[tree_df.child_size > 1]
    cluster_tree = cluster_tree.sort_values(by=['parent', 'lambda_val'], ascending=[True, False])

    def filter_node(cluster_id):

        children = cluster_tree[cluster_tree.parent == cluster_id]

        if not children.empty:
            for _, row in children.iterrows():

                # print(f'Cluster {int(row.child)}, Size: {int(row.child_size)}')

                internal_cluster_id = int(row.child)

                clusters_data = find_final_clusters_in_top_cluster(data, tree, internal_cluster_id, clusters)

                unique_names = set()

                for cluster_id, items in clusters_data.items():
                    # print(f"Internal Cluster ID: {internal_cluster_id} Cluster ID {cluster_id} contains: {items}")
                    for item in items:
                        name = item['name']
                        unique_names.add(name)

                if len(unique_names) <= 4:
                    filter_map = {"names": unique_names, "internal_cluster_id": internal_cluster_id, "cluster_id": cluster_id}
                    filtered_nodes.append(filter_map)
                else:
                    filter_node(row.child)

    roots = cluster_tree[~cluster_tree.parent.isin(cluster_tree.child)]

    for _, row in roots.iterrows():

        filter_node(row.child)

    return filtered_nodes


def main():
    print('Named Entity Cluster')

    nlp = SpacyPipeline.setup_spacy()

    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

    # read in doc list
    # doc_list = []

    with open('../test_output/hp_docs.2.dill', 'rb') as f:
        doc_list = dill.load(f)

    data = []

    entity_id = 0

    for doc in doc_list:

        # SpacyUtils.display_doc(doc)

        entities = doc.spans["entities"]

        for entity in entities:

            entity_type = entity.label_

            if entity_type == 'PERSON':

                entity_id += 1

                entity_start = entity.start
                entity_end = entity.end
                entity_text = entity.text

                print(f"{entity_text} : {entity_start} : {entity_end}")

                sentences = [sent for sent in doc.spans["sentences"] if
                             sent.start <= entity_start and sent.end >= entity_end]

                sentence_text = ""

                if len(sentences) > 0:
                    sentence_text = sentences[0].text
                    print(f"Sentence: {sentence_text}")

                paragraphs = [paragraph for paragraph in doc.spans["paragraphs"] if
                              paragraph.start <= entity_start and paragraph.end >= entity_end]

                paragraph_text = ""

                if len(paragraphs) > 0:
                    paragraph_text = paragraphs[0].text
                    print(f"Paragraph: {paragraph_text}")

                entity_map = {"id": entity_id, "name": entity_text, "context": paragraph_text}

                # entity_map = {"id": entity_id, "name": entity_text, "context": sentence_text}

                data.append(entity_map)

    name_embeddings = model.encode([d['name'] for d in data])

    context_embeddings = model.encode([d['context'] for d in data])

    features = np.hstack((name_embeddings, context_embeddings))

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)

    clusters = clusterer.fit_predict(features)

    for item, cluster in zip(data, clusters):
        print(f"ID: {item['id']}, Name: {item['name']}, Cluster: {cluster}")
        item['cluster_id'] = cluster

    tree = clusterer.condensed_tree_

    print_cluster_hierarchy(data, tree, clusters)

    filtered_nodes = filter_cluster_hierarchy(data, tree, clusters)

    data_length = len(data)

    print(f"Data Length: {data_length}")

    for fn in filtered_nodes:
        print(fn)

    print(f"Filtered Node: {len(filtered_nodes)}")


    # for each document (which is a section)
    # for each named entity mention

    # create node to cluster including:
    # resolved mention label
    # identifier representing mention
    # context as embedding vector
    # which includes surrounding sentences
    # +/- 2 sentences within a paragraph

    # do hierarchical clustering

    # display cluster output

    # assign label to each cluster (use model?)
    # since we're serializing assignments we could assign
    # a better label downstream

    # write out cluster assignments

    # serialize results including cluster assignments:
    # cluster label
    # cluster id
    # cluster parent id (if child)
    # cluster id to mention:
    # doc, paragraph, sentence, mention id
    # use jsonnl output


if __name__ == "__main__":
    main()
