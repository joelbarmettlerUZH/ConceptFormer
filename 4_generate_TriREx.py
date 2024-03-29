import argparse
import csv
import json
import logging
import multiprocessing
import os
import random
import tarfile
from pathlib import Path
from typing import List

import networkx as nx
import requests
from fuzzywuzzy import fuzz
from tqdm import tqdm

from src.Const.prompts import triple_to_trirex_prompt
from src.Datasets.factory import trirex_factory
from src.Datasets.TREx import TREx
from src.Datasets.TRExLite import TRExLite
from src.Datasets.TRExStar import TRExStar
from src.Datasets.TRExStarLite import TRExStarLite

parser = argparse.ArgumentParser(description='Process dataset parameters.')
parser.add_argument('--dataset_name', type=str, default='TriREx',
                    help='Name of the dataset to generate (TriREx or TriRExLite)')
parser.add_argument('--version', type=int, default=1,
                    help='Version number of the dataset')
parser.add_argument('--n_sentences', type=int, default=100,
                    help='Number of sentences to generate per datapoint')
parser.add_argument('--match_threshold', type=int, default=80,
                    help='Fuzzy match threshold for identifying subject and object in sentence')
parser.add_argument('--gpu_indices', nargs='*', type=int, default=[2, 3, 4, 5],
                    help='List of GPU indices to use')
parser.add_argument('--seed', type=int, default=0,
                    help='Seed')
parser.add_argument('--n_processes_per_gpu', type=int, default=2,
                    help='Number of Processes per GPU')
# Parse arguments
args = parser.parse_args()
DATASET_NAME = args.dataset_name
VERSION = args.version
N_SENTENCES = args.n_sentences
GPU_INDICES = args.gpu_indices
SEED = args.seed
MATCH_THRESHOLD = args.match_threshold
N_PROCESSES_PER_GPU = args.n_processes_per_gpu

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

script_directory = Path(__file__).parent
data_directory = script_directory / "data"
weights_directory = data_directory / "mistral" / "7B"
artifacts_directory = data_directory / f"artifacts/{DATASET_NAME}_v{VERSION}"
csv_directory = artifacts_directory / f"csv"
publish_directory = artifacts_directory / f"publish"
output_tar_file_path = publish_directory / f"{DATASET_NAME}_v{VERSION}.tar"

weights_directory.mkdir(parents=True, exist_ok=True)
csv_directory.mkdir(parents=True, exist_ok=True)
publish_directory.mkdir(parents=True, exist_ok=True)

model_name = "mistral-7b-v0.1.Q4_K_M.gguf"
gguf_file_path = weights_directory / model_name
file_url = "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf?download=true"


def download_model():
    """
    Download a file from a URL into a given directory if it does not already exist.
    """
    weights_directory.mkdir(parents=True, exist_ok=True)  # Create directory if it does not exist

    if not gguf_file_path.exists():
        print(f"Downloading {model_name}...")
        response = requests.get(file_url)
        response.raise_for_status()  # Ensure that the download was successful

        with open(gguf_file_path, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded and saved as {gguf_file_path}")


def find_best_match(phrase, sentence):
    words = sentence.split(' ')
    best_score = 0
    best_start = 0
    best_end = 0

    for start in range(len(words)):
        for end in range(start + 1, len(words) + 1):
            slice = ' '.join(words[start:end])
            score = fuzz.ratio(phrase, slice)
            if score > best_score:
                best_score = score
                best_start = start
                best_end = end

    return best_start, best_end, best_score


def process_data_chunk(graphs: List[nx.Graph], gpu_index, SPLIT, n_sentences):
    folder_path = csv_directory / SPLIT
    os.makedirs(folder_path, exist_ok=True)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_index}"

    from llama_cpp import Llama

    llm = Llama(
        model_path=str(gguf_file_path),
        n_gpu_layers=10_000,
        n_threads=4,
        verbose=False,
        n_ctx=1024,
    )

    for G in tqdm(graphs, desc=f"Generating {SPLIT} sentences on GPU {gpu_index}"):
        central_node_id = G.graph['central_node']
        output_path = folder_path / f"{central_node_id}.csv"

        if os.path.isfile(output_path):
            continue

        neighbours = []
        for central_node_id, neighbour_node_id, edge in G.edges(data=True):
            assert central_node_id == G.graph[
                'central_node'], "Graph has wrong format, expect all edges to be from central node"

            central_node_label = G.nodes[central_node_id]['label']
            central_node_rank = G.nodes[central_node_id]['rank']

            relation_id = edge['id']
            relation_label = edge['label']

            neighbour_node_label = G.nodes[neighbour_node_id]['label']
            neighbour_node_rank = G.nodes[neighbour_node_id]['rank']

            neighbours.append({
                "sentence": "",
                "subject_id": central_node_id,
                "subject_label": central_node_label,
                "subject_rank": central_node_rank,
                "subject_boundary_start": -1,
                "subject_boundary_end": -1,
                "predicate_id": relation_id,
                "predicate_label": relation_label,
                "object_id": neighbour_node_id,
                "object_label": neighbour_node_label,
                "object_rank": neighbour_node_rank,
                "object_boundary_start": -1,
                "object_boundary_end": -1,
            })

        neighbours = sorted(neighbours, key=lambda x: x['object_rank'], reverse=True)[:n_sentences]

        for neighbour in neighbours:
            try:
                prompt = triple_to_trirex_prompt(
                    neighbour['subject_label'],
                    neighbour['predicate_label'],
                    neighbour['object_label'],
                )
                output = llm(
                    prompt,
                    max_tokens=256,
                    stop=[".", "\n"],
                    temperature=0.2,
                    echo=False
                )
            except RuntimeError as e:
                print(e)
                continue

            sentence = output['choices'][0]['text']

            if len(sentence) < 2:
                continue

            if sentence[0] in ['"', "'"]:
                sentence = sentence[1:]
            if sentence[-1] in ['"', "'", "."]:
                sentence = sentence[:-1]
            if sentence[-1] in ['"', "'", "."]:
                sentence = sentence[:-1]

            l_sent = sentence.lower()
            l_sub = neighbour['subject_label'].lower()
            l_pred = neighbour['predicate_label'].lower()
            l_obj = neighbour['object_label'].lower()

            preserves_by = " by" not in l_pred or " by " in l_sent

            sub_start, sub_end, sub_score = find_best_match(l_sub, l_sent)
            obj_start, obj_end, obj_score = find_best_match(l_obj, l_sent)

            if sub_score >= MATCH_THRESHOLD and obj_score >= MATCH_THRESHOLD and sub_end < obj_start and preserves_by:
                logging.debug(
                    f"✅ [{neighbour['subject_label']}|{neighbour['predicate_label']}|{neighbour['object_label']}]->{sentence}")

                # Convert word indexes to character indexes
                l_sub_idx = len(' '.join(l_sent.split()[:sub_start])) + (1 if sub_start > 0 else 0)
                l_obj_idx = len(' '.join(l_sent.split()[:obj_start])) + (1 if obj_start > 0 else 0)

                neighbour['sentence'] = sentence
                neighbour['subject_boundary_start'] = l_sub_idx
                neighbour['subject_boundary_end'] = l_sub_idx + len(' '.join(l_sent.split()[sub_start:sub_end]))
                neighbour['object_boundary_start'] = l_obj_idx
                neighbour['object_boundary_end'] = l_obj_idx + len(' '.join(l_sent.split()[obj_start:obj_end]))
            else:
                logging.debug(
                    f"❌ {neighbour['subject_label']}->{neighbour['predicate_label']}->{neighbour['object_label']}|{sentence}")
        neighbours = [neighbour for neighbour in neighbours if neighbour.get('sentence')]

        logger = logging.getLogger(__name__)
        level_numeric = logger.getEffectiveLevel()

        if level_numeric <= 10:
            for neighbour in neighbours:
                subject_info = f"({neighbour['subject_label']},ID={neighbour['subject_id']},rank={neighbour['subject_rank']})"
                object_info = f"({neighbour['object_label']},ID={neighbour['object_id']},rank={neighbour['object_rank']})"
                debug_str = neighbour['sentence'][:]
                debug_str = (
                    f"{debug_str[:neighbour['object_boundary_start']]}"
                    f"[{debug_str[neighbour['object_boundary_start']:neighbour['object_boundary_end']]}]"
                    f"({object_info})"
                    f"{debug_str[neighbour['object_boundary_end']:]}"
                )
                debug_str = (
                    f"{debug_str[:neighbour['subject_boundary_start']]}"
                    f"[{debug_str[neighbour['subject_boundary_start']:neighbour['subject_boundary_end']]}]"
                    f"({subject_info})"
                    f"{debug_str[neighbour['subject_boundary_end']:]}"
                )
                debug_str = f"{neighbour['subject_id']}->{neighbour['predicate_id']}({neighbour['predicate_label']})->{neighbour['object_id']}|{debug_str}"
                logging.debug(debug_str)

        with open(output_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=["sentence", "subject_id", "subject_label", "subject_rank",
                                                   "subject_boundary_start", "subject_boundary_end", "predicate_id",
                                                   "predicate_label", "object_id", "object_label", "object_rank",
                                                   "object_boundary_start", "object_boundary_end"])
            writer.writeheader()
            writer.writerows(neighbours)


def uri_to_id(uri: str) -> str:
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


if __name__ == "__main__":
    if DATASET_NAME == "TriRExLite":
        trex_builder = TRExLite()
        trex_star_builder = TRExStarLite()
    elif DATASET_NAME == "TriREx":
        trex_builder = TREx()
        trex_star_builder = TRExStar()
    else:
        raise ValueError(f"Unknown dataset {DATASET_NAME}")

    if not trex_builder.info.splits:
        trex_builder.download_and_prepare()

    if not trex_star_builder.info.splits:
        trex_star_builder.download_and_prepare()

    download_model()
    multiprocess_gpu_indices = [element for element in GPU_INDICES for _ in range(N_PROCESSES_PER_GPU)]

    graphs = {}
    for datapoint in tqdm(trex_star_builder.as_dataset(split="all"), desc="Loading graphs"):
        data = json.loads(datapoint['json'])
        graphs[datapoint['entity']] = nx.node_link_graph(data)

    for SPLIT in ['train', 'validation', 'test']:

        trex_datapoints = trex_builder.as_dataset(split=SPLIT)
        n_datapoints = len(trex_datapoints)

        split_graphs = []
        for datapoint in tqdm(trex_datapoints, desc="Selecting graphs"):
            graph = graphs.get(uri_to_id(datapoint['docid']))
            if graph:
                split_graphs.append(graph)
        chunk_size = len(split_graphs) // len(multiprocess_gpu_indices)

        random.shuffle(split_graphs)

        processes = []
        for i, gpu_index in enumerate(multiprocess_gpu_indices):
            start_idx = i * chunk_size
            end_idx = None if i == len(multiprocess_gpu_indices) - 1 else (i + 1) * chunk_size
            chunk = split_graphs[start_idx:end_idx]
            p = multiprocessing.Process(target=process_data_chunk, args=(chunk, gpu_index, SPLIT, N_SENTENCES))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    create_output_tar()

    train_dataset, validation_dataset, test_dataset = trirex_factory(DATASET_NAME)

    print(f"{DATASET_NAME}:train", len(train_dataset), train_dataset[0])
    print(f"{DATASET_NAME}:validation", len(validation_dataset), validation_dataset[0])
    print(f"{DATASET_NAME}:test", len(test_dataset), test_dataset[0])
