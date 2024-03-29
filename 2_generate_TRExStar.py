import argparse
import json
import logging
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import networkx as nx
from dotenv import load_dotenv
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from GraphQueryEngine.SparqlEngine import fetch_neighbors, extract_entity_id, get_pagerank_map
from src.Datasets.TREx import TREx
from src.Datasets.TRExLite import TRExLite
from src.Datasets.factory import trex_star_factory

load_dotenv()

# Set up argument parser
parser = argparse.ArgumentParser(description='Process dataset name and version.')
parser.add_argument('--dataset_name', type=str, default='TRExStar',
                    help='Name of the dataset to generate (TRExStar or TRExStarLite)')
parser.add_argument('--version', type=int, default=1,
                    help='Version number of the dataset')
parser.add_argument('--edge_limit', type=int, default=100,
                    help='Maximum number of neighbours per entity')

# Parse arguments
args = parser.parse_args()
DATASET_NAME = args.dataset_name
VERSION = args.version
EDGE_LIMIT = args.edge_limit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


script_directory = Path(__file__).parent
data_directory = script_directory / "data"
artifacts_directory = data_directory / f"artifacts/{DATASET_NAME}_v{VERSION}"
json_directory = artifacts_directory / f"json"
blacklist_file_path = artifacts_directory / f"blacklist.txt"
publish_directory = artifacts_directory / f"publish"
output_tar_file_path = publish_directory / f"{DATASET_NAME}_v{VERSION}.tar"

json_directory.mkdir(parents=True, exist_ok=True)
publish_directory.mkdir(parents=True, exist_ok=True)


def get_all_trex_entities():
    """
    Returns a list of all TREx entity IDs mentioned (as main entities or in triplet relations)
    :return: List of trex entities mentioned
    """
    if DATASET_NAME == "TRExStarLite":
        builder = TRExLite()
    elif DATASET_NAME == "TRExStar":
        builder = TREx()
    else:
        raise ValueError(f"Unknown dataset {DATASET_NAME}")

    if not builder.info.splits:
        builder.download_and_prepare()

    train_dataset = builder.as_dataset(split='train')
    test_dataset = builder.as_dataset(split='test')
    validation_dataset = builder.as_dataset(split='validation')

    # Graphs are only used for lookup, not for for training / testing directly, as their nodes do (and must!) overlap
    datapoints = ConcatDataset([train_dataset, test_dataset, validation_dataset])
    entities = set()

    for data in tqdm(datapoints, desc='Detecting entities'):
        entities.add(extract_entity_id(data["uri"]))
        for entity in data["entities"]:
            entities.add(extract_entity_id(entity["uri"]))
        for triple in data["triples"]:
            entities.add(extract_entity_id(triple["subject"]["uri"]))
            entities.add(extract_entity_id(triple["object"]["uri"]))

    return [entity for entity in entities if entity[0] == "Q"]


def get_blacklist() -> List[str]:
    """
    Some entities from trex might not be in our wikidata dump or their sparql query might fail. We note
    these entities in a blacklist such that we don't retry them over and over again when we restart the script
    :return: List of blacklisted entities
    """
    if not blacklist_file_path.exists():
        return []
    with open(blacklist_file_path, "r") as bl_file:
        return bl_file.read().splitlines()


def process_entity(pageranks: Dict[str, float], entity_id: str):
    """
    Processes an entity by saving its star subgraph
    :param pageranks: Pagerank for every Entity.
    :param entity_id: ID of the central entity
    :return:
    """
    entity_path = json_directory / f"{entity_id}.json"
    if entity_path.exists():
        return

    if entity_id in blacklist:
        return

    G = fetch_neighbors(pageranks, entity_id, EDGE_LIMIT)

    if not G:
        with open(blacklist_file_path, "a") as bl_file:
            bl_file.write(f"{entity_id}\n")
        return

    json_data = nx.node_link_data(G)
    with open(entity_path, 'w') as f:
        json.dump(json_data, f, indent=4)


def create_output_tar():
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


if __name__ == "__main__":
    entities = get_all_trex_entities()
    pageranks = get_pagerank_map()
    blacklist = get_blacklist()

    # Define the number of threads you want to use (for instance, 10)
    num_threads = 8
    futures = []

    # Use ThreadPoolExecutor to parallelize the fetching of neighbors
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        progress_bar = tqdm(total=len(entities), desc="Fetching neighbors")
        for entity_id in entities:
            futures.append(executor.submit(process_entity, pageranks, entity_id))

        for future in as_completed(futures):
            future.result()  # we can retrieve the result if needed
            progress_bar.update(1)

        progress_bar.close()

    create_output_tar()
    dataset = trex_star_factory(DATASET_NAME)

    print(f"{DATASET_NAME}:all", len(dataset))
    print(dataset[0]['entity'])
    print(nx.node_link_graph(json.loads(dataset[0]['json'])))
