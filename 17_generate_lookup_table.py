import argparse
import logging
import multiprocessing
import random
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy
import numpy as np
import torch
from tqdm import tqdm

from src.Datasets.factory import trex_star_graphs_factory
from src.GraphAligner.BigGraphAligner import BigGraphAligner
from src.Config.train_sentences_config import TrainSentencesConfig, gpt2_n_neighbors_search_nested, \
    gpt2_n_neighbors_search_lite_nested, gpt2_n_neighbors_search_dynamic
from src.LLM.factory import llm_factory
from src.Model.GraphAttentionEmbedder.GraphAttentionEmbedder import GraphAttentionEmbedder

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
numpy.random.seed(SEED)
random.seed(SEED)

# Set up argument parser
parser = argparse.ArgumentParser(description='Process dataset name and version.')
parser.add_argument('--gpu_indices', nargs='*', type=int, default=[1],
                    help='List of GPU indices to use')
parser.add_argument('--n_processes_per_gpu', type=int, default=1,
                    help='Number of Processes per GPU')

args = parser.parse_args()
GPU_INDICES = args.gpu_indices
N_PROCESSES_PER_GPU = args.n_processes_per_gpu

script_directory = Path(__file__).parent
data_directory = script_directory / "data"
output_directory = data_directory / f"output/ConceptFormer/"
output_directory.mkdir(parents=True, exist_ok=True)


def main(config: TrainSentencesConfig, graphs: Dict[str, nx.DiGraph], gpu=0):
    device = torch.device(f"cuda:{gpu}")

    publish_directory = output_directory / f"{config.num_pseudo_words}_context_vectors"
    publish_directory.mkdir(parents=True, exist_ok=True)

    llm = llm_factory(
        config.embedding_llm_type,
        config.embedding_llm_name,
        batch_size=1,
        device=device,
        bits=config.quanization
    )

    graph_aligner = BigGraphAligner(llm, graphs, config.graph_dataset_name, use_untrained=True)

    graph_embedder = GraphAttentionEmbedder.from_config(config, llm)

    if not config.trained_path.exists():
        print("Skipping untrained ConceptFormer")
        return

    graph_embedder.load_state_dict(torch.load(config.trained_path, map_location=f'cuda:{gpu}'))
    graph_embedder = graph_embedder.to(device)
    graph_embedder.eval()

    subject_graph_map = list(graphs.items())
    random.shuffle(subject_graph_map)

    for subject_id, G in tqdm(subject_graph_map, desc="SentenceFormer Lookup Table"):
        subject_file_path = publish_directory / f"{subject_id}.npy"

        if subject_file_path.exists() or subject_id not in graph_aligner.entity_index:
            logging.info(f"Embedding for subject_id {subject_id} already exists. Skipping...")
            continue

        central_node_embedding = graph_aligner.node_embedding_batch([subject_id]).unsqueeze(0)

        neighbour_ids, edge_ids, ranks = [], [], []
        for central_node_id, neighbour_node_id, edge in G.edges(data=True):
            edge_ids.append(edge['id'])
            neighbour_ids.append(neighbour_node_id)
            ranks.append(G.nodes[neighbour_node_id]['rank'])

        if len(neighbour_ids) == 0:
            continue

        combined = zip(neighbour_ids, edge_ids, ranks)
        sorted_combined = sorted(combined, key=lambda x: x[2], reverse=True)
        neighbour_ids, edge_ids, _ = zip(*sorted_combined)
        neighbour_ids, edge_ids = list(neighbour_ids), list(edge_ids)

        node_embeddings = graph_aligner.node_embedding_batch(neighbour_ids).unsqueeze(0)
        edge_embeddings = graph_aligner.edge_embedding_batch(edge_ids).unsqueeze(0)

        graph_embeddings = graph_embedder(central_node_embedding, node_embeddings, edge_embeddings)

        # Save the embedding as a numpy array
        np.save(subject_file_path, graph_embeddings.cpu().detach().numpy())

def process_function(config_queue, graphs, gpu):
    while True:
        try:
            config = config_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break

        print(f"Processing config on GPU {gpu}")
        main(config, graphs, gpu)


if __name__ == "__main__":
    multiprocess_gpu_indices = [element for element in GPU_INDICES for _ in range(N_PROCESSES_PER_GPU)]

    configs = gpt2_n_neighbors_search_dynamic

    # Create a multiprocessing queue and add all configurations to it
    config_queue = multiprocessing.Queue()
    for config in configs:
        config_queue.put(config)

    graphs = trex_star_graphs_factory(configs[0].graph_dataset_name)

    processes = []
    for gpu in multiprocess_gpu_indices:
        p = multiprocessing.Process(target=process_function, args=(config_queue, graphs, gpu))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
