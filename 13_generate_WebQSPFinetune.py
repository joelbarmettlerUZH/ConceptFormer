import argparse
import csv
import json
import logging
import random
import tarfile
from pathlib import Path

import networkx as nx
from tqdm import tqdm

from Datasets.factory import benchmark_base_factory, web_qsp_factory, web_qsp_finetune_factory
from GraphQueryEngine.SparqlEngine import mid_to_qid, get_pagerank_map, fetch_neighbors

parser = argparse.ArgumentParser(description='Process dataset name and version.')
parser.add_argument('--version', type=int, default=1,
                    help='Version number of the dataset')


args = parser.parse_args()
DATASET_NAME = "WebQSPFinetune"
VERSION = args.version

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

endpoint_url = "https://query.wikidata.org/bigdata/namespace/wdq/sparql"

script_directory = Path(__file__).parent
data_directory = script_directory / "data"
web_qsp__directory = data_directory / f"benchmarks/WebQSP"

artifacts_sentence_directory = data_directory / f"artifacts/WebQSPFinetuneSentences_v{VERSION}"
csv_sentence_directory = artifacts_sentence_directory / f"csv"
publish_sentence_directory = artifacts_sentence_directory / f"publish"
output_tar_sentence_file_path = publish_sentence_directory / f"WebQSPFinetuneSentences_v{VERSION}.tar"
csv_sentence_directory.mkdir(parents=True, exist_ok=True)
publish_sentence_directory.mkdir(parents=True, exist_ok=True)

artifacts_star_directory = data_directory / f"artifacts/WebQSPFinetuneStar_v{VERSION}"
json_star_directory = artifacts_star_directory / f"json"
publish_star_directory = artifacts_star_directory / f"publish"
output_tar_star_file_path = publish_star_directory / f"WebQSPFinetuneStar_v{VERSION}.tar"
json_star_directory.mkdir(parents=True, exist_ok=True)
publish_star_directory.mkdir(parents=True, exist_ok=True)

def to_trex_format(datapoint: dict, G: nx.Graph):
    annotation = datapoint["Parses"][0]

    central_node_id = G.graph['central_node']
    central_node = G.nodes[central_node_id]
    central_node_label = central_node.get('label')
    central_node_rank = central_node.get('rank')

    answers = []

    for answer in annotation["Answers"]:
        if answer["AnswerType"] != "Entity":
            continue

        entity_id = mid_to_qid(answer['AnswerArgument'])
        neighbor_node = G.nodes.get(entity_id)

        if neighbor_node is None:
            continue

        object_label = neighbor_node.get('label')
        object_rank = neighbor_node.get('rank')

        if not neighbor_node:
            continue

        neighbor_edge = G[central_node_id].get(entity_id)
        if not neighbor_edge:
            continue

        sentence = f"Question: {datapoint['RawQuestion']}\nAnswer: {object_label}."

        try:
            subject_boundary_start = sentence.index(annotation["PotentialTopicEntityMention"])
            subject_boundary_end = subject_boundary_start + len(annotation["PotentialTopicEntityMention"])
        except ValueError:
            continue

        predicate_id = neighbor_edge.get('id')
        predicate_label = neighbor_edge.get('label')

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
        "object_id": entity_id,
        "object_label": object_label,
        "object_rank": object_rank,
        "object_boundary_start": object_boundary_start,
        "object_boundary_end": object_boundary_end,
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

    with open(web_qsp__directory / "WebQSP.train.json", "r") as f:
        web_qsp_train_datapoints = json.load(f)["Questions"]
        random.shuffle(web_qsp_train_datapoints)

    with open(web_qsp__directory / "WebQSP.test.json", "r") as f:
        web_qsp_test_datapoints = json.load(f)["Questions"]
        random.shuffle(web_qsp_test_datapoints)

    star_folder_path = json_star_directory
    star_folder_path.mkdir(parents=True, exist_ok=True)

    splits = {
        'train': web_qsp_train_datapoints,
        'test': web_qsp_test_datapoints,
    }

    for split, web_qsp_datapoints in splits.items():

        sentence_folder_path = csv_sentence_directory / split
        sentence_folder_path.mkdir(parents=True, exist_ok=True)

        for datapoint in tqdm(web_qsp_datapoints, desc=f"Generating WebQSPFinetune Dataset - {split} split"):
            if len(datapoint['Parses']) != 1:
                continue

            mid = datapoint['Parses'][0]['TopicEntityMid']
            if mid is None:
                continue

            question_id = datapoint['QuestionId']

            output_path = sentence_folder_path / f"{question_id}.csv"
            if output_path.exists():
                continue

            entity_id = mid_to_qid(mid)
            if not entity_id:
                continue

            G = fetch_neighbors(pageranks, entity_id, edge_limit=10_000)
            if not G:
                continue

            star_entity_path = json_star_directory / f"{entity_id}.json"
            graph_json_data = nx.node_link_data(G)
            with open(star_entity_path, 'w') as f:
                json.dump(graph_json_data, f, indent=4)

            sentences = to_trex_format(datapoint, G)

            with open(output_path, "w") as f:
                writer = csv.DictWriter(f, fieldnames=["sentence", "subject_id", "subject_label", "subject_rank",
                                                       "subject_boundary_start", "subject_boundary_end",
                                                       "predicate_id", "predicate_label", "object_id",
                                                       "object_label", "object_rank", "object_boundary_start",
                                                       "object_boundary_end"])
                writer.writeheader()
                writer.writerows(sentences)

    create_graph_tar(json_star_directory, output_tar_star_file_path)
    create_sentence_tar(csv_sentence_directory, output_tar_sentence_file_path)

    sentence_train_dataset, sentence_test_dataset, graphs = web_qsp_finetune_factory()
    print(f"WebQSPSentences:train", len(sentence_train_dataset), sentence_train_dataset[0])
    print(f"WebQSPSentences:test", len(sentence_test_dataset), sentence_test_dataset[0])
    print(f"WebQSPStar:all", len(graphs))
