import os

import matplotlib.pyplot as plt
import numpy as np
import wandb

from src.Datasets.factory import trex_star_graphs_factory
from plots.CONSTANTS import ALIAS_BY_MODEL_NAME, COLOR_BY_MODEL_NAME, FORMAT_ALIAS_BY_DESC, DATASET_ALIAS

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
    if len(runs) == 0:
        return None
    return runs[0]

def get_textinjection_run(dataset: str, llm: str):
    project = "university-of-zurich/7_eval_basemodel_textinjection"
    runs = api.runs(project,
                    {"$and": [
                        {"state": "finished"},
                        {"config.dataset": dataset},
                        {"config.llm": llm},
                    ]})
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

    runmap = {
        'baseline': get_baseline_run,
        'graph textification': get_textinjection_run,
    }

    for dataset in ['TriREx', 'TRExBite']:
        for k in [1, 5, 10]:
            for runtype, get_run_fn in runmap.items():
                plt.figure(figsize=(12, 6), dpi=150)
                n_llms = len(ALIAS_BY_MODEL_NAME)
                bar_width = 0.8 / n_llms
                x_values = np.arange(1, batch_size + 1)

                for i, (llm, alias) in enumerate(list(ALIAS_BY_MODEL_NAME.items())[::-1]):
                    run = get_run_fn(dataset, llm)
                    if not run:
                        continue
                    data = table_to_data(get_examples_table(run, "examples"), graphs, batch_size, k)
                    plt.bar(x_values - 0.25 + i * bar_width, data, bar_width, label=alias, color=COLOR_BY_MODEL_NAME[llm], alpha=.99, hatch=("//" if runtype == "graph textification" else None))

                plt.xlabel('Number of Neighbours')
                plt.ylabel(f'Hits@{k}')
                # plt.title(f'{runtype.title()} Evaluation - {DATASET_ALIAS[dataset]}')

                plt.xlim(0.5, batch_size + 0.6)
                plt.xticks(x_values, [f"{batch_size * i + 1}-{batch_size * i + batch_size}" for i in range(batch_size)])

                plt.ylim(0, 1)
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(ALIAS_BY_MODEL_NAME.items()) // 2)
                plt.grid(color='grey', linestyle='-', alpha=0.5)

                plt.tight_layout()
                plt.savefig(os.path.join("output", f"2_eval_{runtype.replace(' ', '_')}_hits_at_{k}_{dataset}.png"), bbox_inches='tight')
                plt.savefig(os.path.join("output", f"2_eval_{runtype.replace(' ', '_')}_hits_at_{k}_{dataset}.pdf"), bbox_inches='tight')

                plt.show()


if __name__ == "__main__":
    plot()
