import os

import numpy as np
from matplotlib.lines import Line2D

import wandb
import matplotlib.pyplot as plt

from src.Datasets.factory import trex_star_graphs_factory
from plots.CONSTANTS import ALIAS_BY_MODEL_NAME, COLOR_BY_MODEL_NAME, DATASET_ALIAS, COLOR_BY_CONCEPTFORMER

PSEUDOWORDS = [1, 2, 3, 4, 5, 10, 15, 20]

api = wandb.Api()
plt.rcParams.update({'font.size': 16})


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

def get_sentenceformer_run(num_pseudo_words: int):
    project = "university-of-zurich/15_evaluate_lookup_table"
    name = f"ConceptFormer-{num_pseudo_words}"
    print(name)
    runs = api.runs(project, {"display_name": name})
    if len(runs) == 0:
        return None
    return runs[0]

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

def plot(batch_size=10):
    wandb.login()
    graphs = trex_star_graphs_factory('TRExStar')
    model_name = ALIAS_BY_MODEL_NAME["gpt2"]

    for dataset_name in ['TriREx', 'TRExBite']:
        for k in [1, 5, 10]:
            baseline_run = get_baseline_run(dataset_name, 'gpt2')
            textinjection_run = get_textinjection_run(dataset_name, 'gpt2')

            baseline_data = table_to_data(get_examples_table(baseline_run, "examples"), graphs, batch_size, k)
            textinjection_data = table_to_data(get_examples_table(textinjection_run, "examples"), graphs, batch_size, k)

            index = np.arange(1, batch_size + 1)

            plt.figure(figsize=(16, 6), dpi=150)
            total_width = 0.8  # Total width for each group of bars
            gap_width = 0.02  # Gap width between bars
            bar_width = (total_width - (len(PSEUDOWORDS) - 1) * gap_width) / len(PSEUDOWORDS)

            for i, num_pseudo_words in enumerate(PSEUDOWORDS):
                bar_positions = index + i * (bar_width + gap_width)
                sentenceformer_run = get_sentenceformer_run(num_pseudo_words)
                if sentenceformer_run is None:
                    continue
                sentenceformer_data = table_to_data(get_examples_table(sentenceformer_run, f"{dataset_name.lower()}_test_examples"), graphs, batch_size, k)
                color = COLOR_BY_CONCEPTFORMER[num_pseudo_words] if num_pseudo_words > 0 else COLOR_BY_MODEL_NAME["gpt2"]
                plt.bar(bar_positions, sentenceformer_data, bar_width, label=f"{model_name} + CF-{num_pseudo_words}", color=color, hatch="..", alpha=.99)

            for i in range(batch_size):
                start = i + 0.95
                end = start + total_width
                plt.hlines(y=baseline_data[i], xmin=start,
                           xmax=end, color='black', linestyle='solid', linewidth=3)

            for i in range(batch_size):
                start = i + 0.95
                end = start + total_width
                plt.hlines(y=textinjection_data[i], xmin=start,
                           xmax=end, color='black', linestyle='dotted', linewidth=3)

            solid_line = Line2D([0], [0], color='black', linewidth=3, linestyle='solid', label=model_name)
            dotted_line = Line2D([0], [0], color='black', linewidth=3, linestyle='dotted', label=f"{model_name} + G-RAG")

            plt.xlabel(f"Number of Neighbours")
            plt.ylabel(f"Hits@{k}")
            plt.xlim(0.5, batch_size + 1)
            plt.ylim(0, 1)
            plt.xticks(index + total_width / 2, [f"{batch_size * i + 1}-{batch_size * i + batch_size}" for i in range(batch_size)])
            plt.grid(color='grey', linestyle='-', alpha=0.5)
            plt.legend(handles=[solid_line, dotted_line] + plt.gca().get_legend_handles_labels()[0], loc='upper center',
                       bbox_to_anchor=(0.5, -0.2), ncol=3)
            # plt.title(f"f{model_name} Baseline vs. Graph Textification vs. ConceptFormer on {DATASET_ALIAS[dataset_name]}")

            plt.savefig(os.path.join("output", f"5_sentenceformer_comparison_{dataset_name}_hits_at_{k}_{batch_size}_batch.png"), bbox_inches='tight')
            plt.savefig(os.path.join("output", f"5_sentenceformer_comparison_{dataset_name}_hits_at_{k}_{batch_size}_batch.pdf"), bbox_inches='tight')
            plt.show()


if __name__ == "__main__":
    plot(batch_size=10)
