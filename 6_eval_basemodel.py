import argparse
import logging
import multiprocessing

import torch
import wandb
from tqdm import tqdm

from src.Datasets.factory import rex_raw_factory, web_qsp_factory
from src.LLM.GPT2 import GPT2
from src.LLM.Llama2 import Llama2

parser = argparse.ArgumentParser(description='Process dataset parameters.')
parser.add_argument('--dataset_name', type=str, default='WebQSP',
                    help='Name of the dataset to evaluate (TriREx, TriRExLite, TRExBite, TRExBiteLite, WebQSP or WebQSPLite)')
parser.add_argument('--gpu_indices', nargs='*', type=int, default=[1, 2],
                    help='List of GPU indices to use')
parser.add_argument('--k', type=int, default=50,
                    help='Maximum p@k')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

args = parser.parse_args()
DATASET_NAME = args.dataset_name
GPU_INDICES = args.gpu_indices
k = args.k

configs = [
    lambda device: GPT2(model_name_or_path="gpt2", max_batch_size=1, device=device),
    lambda device: GPT2(model_name_or_path="gpt2-medium", max_batch_size=1, device=device),
    lambda device: GPT2(model_name_or_path="gpt2-large", max_batch_size=1, device=device),
    lambda device: GPT2(model_name_or_path="gpt2-xl", max_batch_size=1, device=device),
    lambda device: Llama2(model_name_or_path="openlm-research/open_llama_3b_v2", max_batch_size=1, device=device),
    lambda device: Llama2(model_name_or_path="TheBloke/Llama-2-7B-GPTQ", max_batch_size=1, device=device, bits=2),
    lambda device: Llama2(model_name_or_path="TheBloke/Llama-2-13B-GPTQ", max_batch_size=1, device=device, bits=2),
]

def main(config_index, gpu):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() and gpu > -1 else "cpu")

    if DATASET_NAME == "WebQSP":
        sentence_test_data, _ = web_qsp_factory()
    else:
        _, _, sentence_test_data = rex_raw_factory(DATASET_NAME)

    llm = configs[config_index](device)

    wandb.init(
        project=f"6_eval_basemodel",
        group="gpt-2" if llm.name.startswith("gpt") else "llama-2",
        name=llm.model_name_or_path,

        # track hyperparameters and run metadata
        config={
            "llm": llm.model_name_or_path,
            "dataset": DATASET_NAME,
        }
    )

    hits = [0 for _ in range(k)]
    misses = [0 for _ in range(k)]
    total_strived_for_k_hits = 0

    examples_table = wandb.Table(columns=[
        "Source",
        "Target",
        "Prediction",
        "k",
        "Target k",
        "Is Hit",
        "Subject",
        "Predicate",
        "Object",
        "Subject ID",
        "Predicate ID",
        "Object ID",
        "Subject Rank",
        "Object Rank",
    ])

    for i in tqdm(range(len(sentence_test_data)), desc=f'Evaluating', unit='sentences'):
        data = sentence_test_data[i]
        sentence = data['sentence']
        strived_for_k = data.get('k', 1)
        _subject = data['subject']['label']
        _predicate = data['predicate']['label']
        _object = data['object']['label']

        object_start = data['object']['boundaries'][0]
        source_text = sentence[:object_start].strip()
        target_text = sentence[object_start:].strip()

        logging.debug(f"{_subject}->{_predicate}->{_object}:{source_text}|{target_text}")

        if not (source_text and target_text):
            continue

        embeddings = llm.early_embedding(source_text)

        try:
            top_k_output = llm.predict_top_k_sequence(embeddings, source_text=source_text, target_text=target_text, k=k)
        except Exception as e:
            logging.warning(str(e))
            continue

        is_top_k = top_k_output['is_top_k']
        target_k = top_k_output['target_k']

        try:
            prediction = llm.predict_sequence(embeddings, n_tokens=len(top_k_output['target_token_ids']))
        except Exception as e:
            logging.warning(str(e))
            continue

        examples_table.add_data(
            source_text,
            target_text,
            prediction['output_string'],
            target_k,
            strived_for_k,
            bool(target_k and target_k <= strived_for_k),
            _subject,
            _predicate,
            _object,
            data['subject']['id'],
            data['predicate']['id'],
            data['object']['id'],
            data['subject']['rank'],
            data['object']['rank'],
        )

        if target_k and target_k <= strived_for_k:
            total_strived_for_k_hits += 1

        if not is_top_k:
            for i in range(len(misses)):
                misses[i] += 1
        else:
            for i in range(target_k - 1):
                misses[i] += 1

            for i in range(target_k - 1, len(misses)):
                hits[i] += 1

    x_values = list(range(1, k + 1))
    y_values = list([hits[i] / (hits[i] + misses[i]) for i in range(k)])

    data = [[x, y] for (x, y) in zip(x_values, y_values)]
    topk_table = wandb.Table(data=data, columns=["x", "y"])
    wandb.log({
        "examples": examples_table,
        "plot": wandb.plot.line(topk_table, "x", "y", title="TopK Hit Ratio"),
    })

    wandb.summary[f'n_sentences'] = hits[0] + misses[0]
    wandb.summary[f'strived_for_k_hit_ratio'] = total_strived_for_k_hits / (hits[0] + misses[0])

    for k_i in [1, 2, 3, 4, 5, 10, 15, 25, 50]:
        wandb.summary[f'k{k_i}'] = hits[k_i - 1] / (hits[k_i - 1] + misses[k_i - 1])

    wandb.finish()


def process_function(config_indices_queue, gpu):
    while True:
        try:
            config_index = config_indices_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break

        print(f"Processing {DATASET_NAME} on device {gpu}")
        main(config_index, gpu)


if __name__ == "__main__":
    devices = GPU_INDICES

    if len(devices) == 1 and devices[0] == -1:
        devices = [-1 for _ in range(multiprocessing.cpu_count())]

    # Create a multiprocessing queue and add all configurations to it
    config_indices_queue = multiprocessing.Queue()
    for config_index in range(len(configs)):
        config_indices_queue.put(config_index)

    processes = []
    for device in devices:
        p = multiprocessing.Process(target=process_function, args=(config_indices_queue, device))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
