import argparse
import multiprocessing
import random
from dataclasses import asdict
from pathlib import Path

import numpy
import numpy as np
import torch
import wandb
from torch.utils.data import Dataset
from tqdm import tqdm

from src.Datasets.factory import trirex_factory, trex_bite_factory, web_qsp_factory
from src.Config.train_sentences_config import TrainSentencesConfig, gpt2_n_neighbors_search, \
    gpt2_n_neighbors_search_lite, gpt2_n_neighbors_search_dynamic
from src.LLM.factory import llm_factory
from src.Model.Trainer.SentenceTrainer import SentenceTrainer

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
numpy.random.seed(SEED)
random.seed(SEED)

# Set up argument parser
parser = argparse.ArgumentParser(description='Process dataset name and version.')
parser.add_argument('--gpu_indices', nargs='*', type=int, default=[0],
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


class CompatLoader(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sentence = self.dataset[idx]
        subject_boundary_start, subject_boundary_end = sentence['subject']['boundaries']
        object_boundary_start, object_boundary_end = sentence['object']['boundaries']
        return {
            'sentence': sentence['sentence'],
            'subject_id': sentence['subject']['id'],
            'subject_label': sentence['subject']['label'],
            'subject_rank': sentence['subject']['rank'],
            'predicate_id': sentence['predicate']['id'],
            'predicate_label': sentence['predicate']['label'],
            'object_id': sentence['object']['id'],
            'object_label': sentence['object']['label'],
            'object_rank': sentence['object']['rank'],
            'subject_boundary_start': subject_boundary_start,
            'subject_boundary_end': subject_boundary_end,
            'object_boundary_start': object_boundary_start,
            'object_boundary_end': object_boundary_end,
        }


def main(config: TrainSentencesConfig, gpu=0, k=50):
    device = torch.device(f"cuda:{gpu}")
    publish_directory = output_directory / f"{config.num_pseudo_words}_context_vectors"

    llm = llm_factory(
        config.embedding_llm_type,
        config.embedding_llm_name,
        batch_size=config.batch_size,
        device=device,
        bits=config.quanization
    )
    trainer = SentenceTrainer(llm, graph_embedder=None, replace_subject=config.replace_subject)
    trainer.to(device)

    run = wandb.init(
        project="15_evaluate_lookup_table",
        name=f"SentenceFormer-{config.num_pseudo_words}",
        config=asdict(config),
    ),

    trirex_train, _, trirex_test = trirex_factory(config.pretrain_dataset_name)
    trex_bite_train, _, trex_bite_test = trex_bite_factory(config.train_dataset_name)

    for test_dataset, prefix in (
            (trex_bite_test, "trexbite_test_"),
            (trirex_test, "trirex_test_"),
            (trirex_train, "trirex_train_"),
            (trex_bite_train, "trexbite_train_"),
    ):
        lookup_table = {}
        for i in tqdm(range(len(test_dataset)), desc=f'Picking relevant entities'):
            data = test_dataset[i]
            subject_id = data['subject']['id']
            subject_file_path = publish_directory / f"{subject_id}.npy"
            try:
                lookup_table[subject_id] = torch.from_numpy(np.load(subject_file_path)).to(device)
            except FileNotFoundError:
                continue

        if lookup_table:
            trainer.evaluate(test_dataset=CompatLoader(test_dataset), k=k, run=run, embedding_lookup_table=lookup_table, prefix=prefix)

    wandb.finish()


def process_function(config_queue, gpu):
    while True:
        try:
            config = config_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break

        print(f"Processing config on GPU {gpu}")
        main(config, gpu)


if __name__ == "__main__":
    multiprocess_gpu_indices = [element for element in GPU_INDICES for _ in range(N_PROCESSES_PER_GPU)]

    configs = gpt2_n_neighbors_search_dynamic

    # Create a multiprocessing queue and add all configurations to it
    config_queue = multiprocessing.Queue()
    for config in configs:
        config_queue.put(config)

    processes = []
    for gpu in multiprocess_gpu_indices:
        p = multiprocessing.Process(target=process_function, args=(config_queue, gpu))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
