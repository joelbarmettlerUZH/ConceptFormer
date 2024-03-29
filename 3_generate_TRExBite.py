import argparse
import csv
import logging
import tarfile
from pathlib import Path

import networkx as nx
from dotenv import load_dotenv
from tqdm import tqdm

from src.Datasets.factory import trex_bite_base_factory, trex_bite_factory

load_dotenv()

# Set up argument parser
parser = argparse.ArgumentParser(description='Process dataset name and version.')
parser.add_argument('--dataset_name', type=str, default='TRExBite',
                    help='Name of the dataset to generate (TRExBite or TRExBiteLite)')
parser.add_argument('--version', type=int, default=1,
                    help='Version number of the dataset')

# Parse arguments
args = parser.parse_args()
DATASET_NAME = args.dataset_name
VERSION = args.version

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

script_directory = Path(__file__).parent
data_directory = script_directory / "data"
artifacts_directory = data_directory / f"artifacts/{DATASET_NAME}_v{VERSION}"
csv_directory = artifacts_directory / f"csv"
publish_directory = artifacts_directory / f"publish"
output_tar_file_path = publish_directory / f"{DATASET_NAME}_v{VERSION}.tar"

csv_directory.mkdir(parents=True, exist_ok=True)
publish_directory.mkdir(parents=True, exist_ok=True)

def find_entity_span(sentence: str, entity: str) -> int:
    """
    For a given sentence and a given entity, returns span of the entity at the beginning of the sentence.
    For example: The sentence "Marie Skłodowska Curie was a Polish and naturalised-French physicist" together
    with the entity "Marie Curie" would return (22) as the entity spans between the index 0 and 22.
    If the entity is not present, function returns -1
    :param sentence: Sentence string
    :param entity: Entity that shall be deteected
    :return: Index of entity, or -1
    """
    relevant_sentence_part = sentence[:len(entity) * 2].lower()  # Only check beginning of sentence
    entity_parts = entity.lower().split(" ")  # We allow character or words between the entity parts (e.g. "Skłodowska")
    last_index = 0
    for part in entity_parts:
        index = relevant_sentence_part.find(part)
        if index > -1:
            last_index = max(last_index, index + len(part))
        else:
            return -1
    return last_index


def uri_to_id(uri: str) -> str | None:
    """
    Resolves the wikidata Q-ID from a wikidata URI
    :param uri: Wikidata URI
    :return: Wikidata Q-ID or None
    """
    if not "wikidata.org" in uri:
        return None
    wikidata_id = uri.split("/")[-1]
    if not wikidata_id.startswith("Q"):
        return None
    return wikidata_id


def extract_valid_text_parts(main_entity_id: str, datapoint: dict, G: nx.Graph):
    """
    For a given TREx datapoint, extract as many valid text parts as possible.
    A text part is valid if:
    - It is the start of a sentence
    - It contains the main entity
    - The graph for the main entity is part of the TRExStar Dataset
    - It contains any neighboring entity of the TRExStar Subgraph
    - The main entity appears before the neighbouring entity
    - The sentence part before the neighbouring entity appears is at most 512 characters long
    :param datapoint: TREx Datapoint
    :param graphs: Map from entity_id strings to corresponding NX Graphs, containing all graphs from TRExStar
    :return: List of valid text parts
    """
    valid_text_parts = []
    sentences = []

    mentioned_entities = datapoint['entities']
    valid_starting_points = []

    # This is needed as the entity at the start of the Wikipedia Text is often overseen by TREx
    entity_at_start = find_entity_span(datapoint['text'], datapoint['title'])
    if entity_at_start > -1:
        valid_starting_points.append({
            "start": 0,
            "entity_boundaries": [0, entity_at_start]
        })

    for main_entity in mentioned_entities:
        # Skip if not main entity
        if uri_to_id(main_entity['uri']) != main_entity_id:
            continue
        main_entity_start, _ = main_entity["boundaries"]
        prepending_text = datapoint["text"][:main_entity_start]
        index_of_last_point = prepending_text.rfind(".")
        if index_of_last_point == -1:
            index_of_last_point = 0
        else:
            index_of_last_point += 2  # "...end of sentence. New start..." -> "New start..." +2 to skip dot and whitespace
        valid_starting_points.append({
            "start": index_of_last_point,
            "entity_boundaries": main_entity["boundaries"]
        })

    # No valid starting pont, happens when main entity was never linked or always extremely late in sentences
    if len(valid_starting_points) == 0:
        return []

    central_node_id = G.graph['central_node']
    central_node = G.nodes[central_node_id]
    central_node_label = central_node.get('label')
    central_node_rank = central_node.get('rank')

    for mentioned_entity in mentioned_entities:
        entity_id = uri_to_id(mentioned_entity['uri'])
        if not entity_id or entity_id == main_entity_id:
            continue

        neighbor_node = G.nodes.get(entity_id)
        if not neighbor_node:
            continue
        neighbor_edge = G[central_node_id][entity_id]

        mentioned_entity_start, mentioned_entity_end = mentioned_entity["boundaries"]
        for starting_point in valid_starting_points:
            text_part_start = starting_point["start"]
            main_entity_start, main_entity_end = starting_point["entity_boundaries"]
            if mentioned_entity_start < text_part_start or mentioned_entity_start < main_entity_end:
                continue
            sentence = datapoint["text"][text_part_start:mentioned_entity_end]
            n_chars = len(sentence)

            # Prevent duplicates
            if sentence in sentences:
                continue

            if n_chars < 512:
                subject_boundary_start = main_entity_start - text_part_start
                subject_boundary_end = main_entity_end - text_part_start

                predicate_id = neighbor_edge.get('id')
                predicate_label = neighbor_edge.get('label')

                object_boundary_start = mentioned_entity_start - text_part_start
                object_boundary_end = mentioned_entity_end - text_part_start

                # Prevent object to be the start of a new sentence, as this is fairly arbitrary
                if "." in sentence[object_boundary_start-3:object_boundary_start]:
                    continue

                logger = logging.getLogger(__name__)
                level_numeric = logger.getEffectiveLevel()

                if level_numeric <= 10:
                    subject_info = f"({central_node_label},ID={main_entity_id},rank={central_node_rank})"
                    object_info = f"({neighbor_node.get('label')},ID={entity_id},rank={neighbor_node.get('rank')})"
                    debug_str = sentence[:]
                    debug_str = (
                        f"{debug_str[:object_boundary_start]}"
                        f"[{debug_str[object_boundary_start:object_boundary_end]}]"
                        f"({object_info})"
                        f"{debug_str[object_boundary_end:]}"
                    )
                    debug_str = (
                        f"{debug_str[:subject_boundary_start]}"
                        f"[{debug_str[subject_boundary_start:subject_boundary_end]}]"
                        f"({subject_info})"
                        f"{debug_str[subject_boundary_end:]}"
                    )
                    debug_str = f"{main_entity_id}->{predicate_id}({predicate_label})->{entity_id}|{debug_str}"
                    logging.debug(debug_str)
                valid_text_parts.append({
                    "sentence": sentence,
                    "subject_id": central_node_id,
                    "subject_label": central_node_label,
                    "subject_rank": central_node_rank,
                    "subject_boundary_start": subject_boundary_start,
                    "subject_boundary_end": subject_boundary_end,
                    "predicate_id": predicate_id,
                    "predicate_label": predicate_label,
                    "object_id": entity_id,
                    "object_label": neighbor_node.get('label'),
                    "object_rank": neighbor_node.get('rank'),
                    "object_boundary_start": object_boundary_start,
                    "object_boundary_end": object_boundary_end,
                })
                sentences.append(sentence)
    return valid_text_parts

def create_output_tar():
    # Check if the csv directory exists
    if not csv_directory.exists():
        raise Exception(f"Directory {csv_directory} does not exist.")

    # Creating a tar file
    with tarfile.open(output_tar_file_path, "w") as tar:
        # Loop through the subdirectories "test", "train", and "validation"
        for subdirectory in ["test", "train", "validation"]:
            subdirectory_path = csv_directory / subdirectory
            if subdirectory_path.exists():
                for file_path in subdirectory_path.glob('*.csv'):
                    # Add each file to the tar, preserving the subdirectory structure
                    tar.add(file_path, arcname=str(file_path.relative_to(csv_directory)))

    print(f"Tar file created at {output_tar_file_path}")


if __name__== "__main__":
    train_dataset, validation_dataset, test_dataset, graphs = trex_bite_base_factory(DATASET_NAME)

    split_map = { 'train': train_dataset, 'validation': validation_dataset, 'test': test_dataset }

    for SPLIT in ['train', 'validation', 'test']:
        folder_path = csv_directory / SPLIT
        folder_path.mkdir(parents=True, exist_ok=True)

        trex_datapoints = split_map[SPLIT]

        for datapoint in tqdm(trex_datapoints, desc=f"Generating {SPLIT} bites"):
            entity_id = uri_to_id(datapoint['docid'])
            if not entity_id:
                continue
            G = graphs.get(entity_id)
            if not G:
                continue
            output_path = folder_path / f"{entity_id}.csv"
            if output_path.exists():
                continue
            sentences = extract_valid_text_parts(entity_id, datapoint, G)
            with open(output_path, "w") as f:
                writer = csv.DictWriter(f, fieldnames=["sentence", "subject_id", "subject_label", "subject_rank",
                                                       "subject_boundary_start", "subject_boundary_end",
                                                       "predicate_id", "predicate_label", "object_id",
                                                       "object_label", "object_rank", "object_boundary_start",
                                                       "object_boundary_end"])
                writer.writeheader()
                writer.writerows(sentences)

    create_output_tar()

    train_dataset, validation_dataset, test_dataset = trex_bite_factory(DATASET_NAME)
    print(f"{DATASET_NAME}:train", len(train_dataset), train_dataset[0])
    print(f"{DATASET_NAME}:validation", len(validation_dataset), validation_dataset[0])
    print(f"{DATASET_NAME}:test", len(test_dataset), test_dataset[0])
