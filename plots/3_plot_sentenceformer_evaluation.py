import os
from functools import lru_cache

import numpy as np
import networkx as nx
import wandb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from src.Datasets.factory import trex_star_graphs_factory
from src.LLM.GPT2 import GPT2
from plots.CONSTANTS import ALIAS_BY_MODEL_NAME, COLOR_BY_MODEL_BASE, COLOR_BY_MODEL_NAME, COLOR_BY_CONCEPTFORMER

api = wandb.Api()

plt.rcParams.update({'font.size': 16})

device = torch.device("cuda:1")
llm = GPT2(model_name_or_path="gpt2", max_batch_size=1, device=device)
PSEUDO_WORDS = [1, 2, 3, 4, 5, 10, 15, 20]

def download_and_read_table(run, table_name:str):
    artifacts = run.logged_artifacts()  # This gets all logged artifacts
    for artifact in artifacts:
        if artifact.name.endswith(f"{table_name}:v0"):
            return artifact.get(table_name)

def get_baseline_run(dataset: str, llm: str):
    project = "university-of-zurich/6_eval_basemodel"
    runs = api.runs(project,
                    {"$and": [
                        {"state": "finished"},
                        {"config.dataset": dataset},
                        {"config.llm": llm},
                    ]})
    return runs[0]

def get_textinjection_run(dataset: str, llm: str):
    project = "university-of-zurich/7_eval_basemodel_textinjection"
    runs = api.runs(project,
                    {"$and": [
                        {"state": "finished"},
                        {"config.dataset": dataset},
                        {"config.llm": llm},
                    ]})
    return runs[0]

def get_text_injection_len(G:nx.Graph) -> int:
    node_labels = []
    edge_labels = []

    for central_node_id, neighbour_node_id, edge in G.edges(data=True):
        neighbour_node_label = G.nodes[neighbour_node_id]['label']
        relation_label = edge['label']
        node_labels.append(neighbour_node_label)
        edge_labels.append(relation_label)

    injection = f"{edge_labels[0]} {node_labels[0]}"
    for node, edge in list(zip(node_labels, edge_labels))[1:]:
        injection = f"{injection}, {edge} {node}"

    injection = f", {injection},  "
    return len(llm.tokenizer.encode(injection))

def get_sentenceformer_run(num_pseudo_words: int):
    project = "university-of-zurich/15_evaluate_lookup_table"
    name = f"ConceptFormer-{num_pseudo_words}"
    print(name)
    runs = api.runs(project, {"display_name": name})
    if len(runs) == 0:
        return None
    return runs[0]

@lru_cache(maxsize=None)
def get_examples_table(run, table_name: str):
    table = download_and_read_table(run, table_name)
    return table

def table_to_data(table, graphs, batch_size, baseline_k) -> np.array:
    num_batches = 100 // batch_size
    neighbour_to_k_table = [[] for _ in range(num_batches)]
    k_index = table.columns.index('k')
    subject_id_index = table.columns.index('Subject ID')
    for datapoint in table.data:
        subject_id = datapoint[subject_id_index]
        k = datapoint[k_index]
        G = graphs.get(subject_id)
        if G is None:
            continue
        n_neighbours = len(G) - 1
        batch_index = min(n_neighbours // batch_size, num_batches-1)
        is_hit = 1 if k and k <= baseline_k else 0
        neighbour_to_k_table[batch_index].append(is_hit)
    for i in range(num_batches):
        neighbour_to_k_table[i] = sum(neighbour_to_k_table[i]) / (len(neighbour_to_k_table[i]) + 1e-10)
    return np.array(neighbour_to_k_table)


def get_token_injection_data(graphs, runs, batch_size):
    num_batches = 100 // batch_size
    token_injection_table = [[] for _ in range(num_batches)]
    for run in runs:
        table = get_examples_table(run, "examples")
        for datapoint in table.data:
            subject_id = datapoint[table.columns.index('Subject ID')]
            G = graphs.get(subject_id)
            if G is None:
                continue
            n_neighbours = len(G) - 1
            batch_index = min(n_neighbours // batch_size, num_batches - 1)
            token_injection_table[batch_index].append(get_text_injection_len(G))
    for i in range(num_batches):
        token_injection_table[i] = np.mean(token_injection_table[i]) if token_injection_table[i] else 0
    return np.array(token_injection_table)



def plot(batch_size=10):
    wandb.login()
    graphs = trex_star_graphs_factory('TRExStar')
    model_name = ALIAS_BY_MODEL_NAME["gpt2"]

    for dataset_name in ['TriREx','TRExBite']:
        for k in [1, 5, 10]:
            baseline_run = get_baseline_run(dataset_name, 'gpt2')
            textinjection_run = get_textinjection_run(dataset_name, 'gpt2')

            baseline_data = table_to_data(get_examples_table(baseline_run, "examples"), graphs, batch_size, k)
            textinjection_data = table_to_data(get_examples_table(textinjection_run, "examples"), graphs, batch_size, k)

            token_injection_textinjection = get_token_injection_data(graphs, [textinjection_run], batch_size)

            x_values = np.arange(1, len(baseline_data) + 1)
            width = 0.25

            for num_pseudo_words in PSEUDO_WORDS:
                sentenceformer_run = get_sentenceformer_run(num_pseudo_words)

                if sentenceformer_run is None:
                    continue

                sentenceformer_data = table_to_data(get_examples_table(sentenceformer_run, f"{dataset_name.lower()}_test_examples"), graphs, batch_size, k)
                token_injection_sentenceformer = np.full_like(sentenceformer_data, num_pseudo_words)

                plt.figure(figsize=(15, 6), dpi=150)
                gs = gridspec.GridSpec(1, 3)  # 1 row, 3 columns

                # First subplot (left), taking up two columns
                ax1 = plt.subplot(gs[0, :2])  # This subplot spans the first two columns
                ax1.bar(x_values - width, baseline_data, width, label=model_name, color="#D60270")
                ax1.bar(x_values, textinjection_data, width, label=f"{model_name} + G-RAG", color="#9B4F96", hatch="//", alpha=.99)
                ax1.bar(x_values + width, sentenceformer_data, width, label=f"{model_name} + CF-{num_pseudo_words}", color="#0038A8", hatch="..", alpha=.99)

                ax1.set_xlabel(f"Number of Neighbours")
                ax1.set_ylabel(f"Hits@{k}")
                ax1.set_xlim(0.5, len(baseline_data) + 0.5)
                ax1.set_ylim(0, 1)
                ax1.set_xticks(x_values)
                ax1.set_xticklabels([f"{batch_size * i + 1}-{batch_size * i + batch_size}" for i in range(len(baseline_data))])
                ax1.grid(color='grey', linestyle='-', alpha=0.5)

                # Second subplot (right), taking up one column
                ax2 = plt.subplot(gs[0, 2])  # This subplot spans the third column
                second_width = 0.33
                ax2.bar(x_values - second_width, token_injection_textinjection, second_width, color="#9B4F96", hatch="//", alpha=.99)
                ax2.bar(x_values, token_injection_sentenceformer, second_width, color="#0038A8", hatch="..", alpha=.99)

                ax2.set_xlabel(f"Number of Neighbours")
                ax2.set_ylabel("Average Tokens Injected")
                ax2.set_xlim(0, len(baseline_data) + 0.5)
                ax2.set_xticks([x_values[0], x_values[-1]])
                ax2.set_xticklabels([0, 100])
                ax2.grid(color='grey', linestyle='-', alpha=0.5)

                # Create a single legend for the figure
                handles, labels = ax1.get_legend_handles_labels()
                plt.figlegend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

                plt.tight_layout()
                plt.savefig(os.path.join("output", f"3_combined_{dataset_name}_hits_at_{k}_sentenceformer-{num_pseudo_words}_{batch_size}.png"), bbox_inches='tight')
                plt.savefig(os.path.join("output", f"3_combined_{dataset_name}_hits_at_{k}_sentenceformer-{num_pseudo_words}_{batch_size}.pdf"), bbox_inches='tight')
                plt.close()


if __name__ == "__main__":
    plot(batch_size=10)
