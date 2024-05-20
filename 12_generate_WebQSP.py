import argparse
import csv
import json
import logging
import tarfile
from pathlib import Path
from typing import List, Dict

import networkx as nx
from tqdm import tqdm

from Datasets.factory import web_qsp_factory
from GraphQueryEngine.SparqlEngine import get_pagerank_map, fetch_neighbors, get_entity_label

parser = argparse.ArgumentParser(description='Process dataset name and version.')
parser.add_argument('--version', type=int, default=1,
                    help='Version number of the dataset')

args = parser.parse_args()
VERSION = args.version

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

script_directory = Path(__file__).parent
data_directory = script_directory / "data"
web_qsp__directory = data_directory / f"benchmarks/WebQSP"

artifacts_sentence_directory = data_directory / f"artifacts/WebQSPSentences_v{VERSION}"
csv_sentence_directory = artifacts_sentence_directory / f"csv"
publish_sentence_directory = artifacts_sentence_directory / f"publish"
output_tar_sentence_file_path = publish_sentence_directory / f"WebQSPSentences_v{VERSION}.tar"
csv_sentence_directory.mkdir(parents=True, exist_ok=True)
publish_sentence_directory.mkdir(parents=True, exist_ok=True)

artifacts_star_directory = data_directory / f"artifacts/WebQSPStar_v{VERSION}"
json_star_directory = artifacts_star_directory / f"json"
publish_star_directory = artifacts_star_directory / f"publish"
output_tar_star_file_path = publish_star_directory / f"WebQSPStar_v{VERSION}.tar"
json_star_directory.mkdir(parents=True, exist_ok=True)
publish_star_directory.mkdir(parents=True, exist_ok=True)


def find_entity_boundaries(sentence: str, token_ids: List[int] | None):
    words = sentence.split()
    try:
        start_index = sum(len(words[i]) + 1 for i in range(token_ids[0]))  # +1 for spaces
        start_index -= 1
        entity_length = sum(len(words[i]) for i in token_ids) + (len(token_ids) - 1)
        end_index = start_index + entity_length
    except IndexError:
        return None

    return [start_index, end_index]

def to_sentence_format(
    pageranks: Dict[str, float],
    utterance: str,
    answer_ids: List[str],
    boundaries: List[int],
    G: nx.Graph,
):
    central_node_id = G.graph['central_node']
    central_node = G.nodes[central_node_id]
    central_node_label = central_node.get('label')
    central_node_rank = central_node.get('rank')

    answers = []

    for answer_id in answer_ids:
        neighbor_node = G.nodes.get(answer_id)

        if neighbor_node:
            object_label = neighbor_node.get('label')
            object_rank = neighbor_node.get('rank')
        else:
            object_label = get_entity_label(answer_id)
            object_rank = pageranks.get(answer_id, 0.5)

        if object_label is None:
            print(f"Entity '{answer_id}' no longer exists - skipping")
            continue

        neighbor_edge = G[central_node_id].get(answer_id)
        if neighbor_edge:
            predicate_id = neighbor_edge.get('id')
            predicate_label = neighbor_edge.get('label')
        else:
            predicate_id = "Unknown"
            predicate_label = "Unknown"

        prefix = 'Question: '
        sentence = f"{prefix}{utterance}\nAnswer: {object_label}."

        subject_boundary_start = boundaries[0] + len(prefix)
        subject_boundary_end = boundaries[1] + len(prefix)

        object_boundary_start = sentence.index(object_label)
        object_boundary_end = object_boundary_start + len(object_label)

        answers.append({
            "sentence": sentence,
            "subject_id": central_node_id,
            "subject_label": central_node_label,
            "subject_rank": central_node_rank,
            "subject_boundary_start": subject_boundary_start,
            "subject_boundary_end": subject_boundary_end,
            "predicate_id": predicate_id,
            "predicate_label": predicate_label,
            "object_id": answer_id,
            "object_label": object_label,
            "object_rank": object_rank,
            "object_boundary_start": object_boundary_start,
            "object_boundary_end": object_boundary_end,
            "k": len(answer_ids),
        })

    return answers

def create_sentence_tar(data_directory: Path, output_tar_file_path: Path):
    # Check if the csv directory exists
    if not data_directory.exists():
        raise Exception(f"Directory {data_directory} does not exist.")

    # Creating a tar file
    with tarfile.open(output_tar_file_path, "w") as tar:
        # Loop through the subdirectories "test", "train", and "validation"
        for subdirectory in ["test", "train", "validation"]:
            subdirectory_path = data_directory / subdirectory
            if subdirectory_path.exists():
                for file_path in subdirectory_path.glob('*.csv'):
                    # Add each file to the tar, preserving the subdirectory structure
                    tar.add(file_path, arcname=str(file_path.relative_to(data_directory)))

    print(f"Tar file created at {output_tar_file_path}")


def create_graph_tar(json_directory:Path, output_tar_file_path:Path):
    """
    Saves the generated json files into a tar that is later used by HF Dataset
    :return:
    """
    # Check if the json directory exists
    if not json_directory.exists():
        raise Exception(f"Directory {json_directory} does not exist.")

    # Creating a tar file
    with tarfile.open(output_tar_file_path, "w") as tar:
        for file_path in json_directory.glob('*.json'):
            tar.add(file_path, arcname=file_path.name)

    print(f"Tar file created at {output_tar_file_path}")


if __name__== "__main__":
    pageranks = get_pagerank_map()

    with open(web_qsp__directory / "webqsp.examples.test.wikidata.json", "r") as f:
        web_qsp_datapoints = json.load(f)

    star_folder_path = json_star_directory
    star_folder_path.mkdir(parents=True, exist_ok=True)

    sentence_folder_path = csv_sentence_directory / "test"
    sentence_folder_path.mkdir(parents=True, exist_ok=True)

    for datapoint in tqdm(web_qsp_datapoints, desc=f"Generating WebQSP benchmark"):
        if len(datapoint['entities']) != 1:
            print("FOUND MULTIPLE LINKED ENTITIES")
            continue

        entity = datapoint['entities'][0]
        if len(entity['linkings']) > 1:
            print("FOUND MULTIPLE LINKINGS")
            continue

        token_ids = entity['token_ids']
        if len(token_ids) == 0:
            print("NO LINKED ENTITY")
            continue

        question_id = datapoint['questionid']
        entity_id = entity['linkings'][0][0]
        answer_ids = list(set(datapoint['answers']))
        sentence = datapoint['utterance']
        boundaries = find_entity_boundaries(sentence, token_ids)

        if boundaries is None:
            continue

        output_path = sentence_folder_path / f"{question_id}.csv"
        if output_path.exists():
            continue

        G = fetch_neighbors(pageranks, entity_id, edge_limit=10_000)
        if not G:
            print("Graph could not be fetched")
            continue

        star_entity_path = json_star_directory / f"{entity_id}.json"
        graph_json_data = nx.node_link_data(G)
        with open(star_entity_path, 'w') as f:
            json.dump(graph_json_data, f, indent=4)

        sentences = to_sentence_format(pageranks, sentence, answer_ids, boundaries, G)

        with open(output_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=["sentence", "subject_id", "subject_label", "subject_rank",
                                                   "subject_boundary_start", "subject_boundary_end",
                                                   "predicate_id", "predicate_label", "object_id",
                                                   "object_label", "object_rank", "object_boundary_start",
                                                   "object_boundary_end", "k"])
            writer.writeheader()
            writer.writerows(sentences)

    create_graph_tar(json_star_directory, output_tar_star_file_path)
    create_sentence_tar(csv_sentence_directory, output_tar_sentence_file_path)

    sentence_test_dataset, graphs = web_qsp_factory()
    print(f"WebQSPSentences:test", len(sentence_test_dataset), sentence_test_dataset[0])
    print(f"WebQSPStar:all", len(graphs))
