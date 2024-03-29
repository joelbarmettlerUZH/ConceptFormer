import os

import matplotlib.pyplot as plt
import wandb

from plots.CONSTANTS import ALIAS_BY_MODEL_NAME, COLOR_BY_MODEL_NAME, COLOR_BY_CONCEPTFORMER

PSEUDOWORDS = [1, 2, 3, 4, 5, 10, 15, 20]

api = wandb.Api()

plt.rcParams.update({'font.size': 16})

def get_sentenceformer_run(num_pseudo_words: int):
    project = "university-of-zurich/15_evaluate_lookup_table"
    name = f"ConceptFormer-{num_pseudo_words}"
    print(name)
    runs = api.runs(project, {"display_name": name})
    if len(runs) == 0:
        return None
    return runs[0]

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

def plot():
    wandb.login()

    for dataset in ['TriREx', 'TRExBite']:
        for testtrain in ['test', 'train']:
            for k in [1, 5, 10]:
                plt.figure(figsize=(12, 8), dpi=150)

                x_idx = 0

                for i, (llm, alias) in enumerate(list(ALIAS_BY_MODEL_NAME.items())[::-1]):
                    run = get_baseline_run(dataset, llm)
                    if run is None:
                        x_idx += 1.2
                        continue
                    x_vals = [x_idx]
                    y_vals = [run.summary.get(f'k{k}', 0)]
                    plt.bar(x_vals, y_vals, 1, label=alias, color=COLOR_BY_MODEL_NAME[llm])
                    x_idx += 1.2

                x_idx += 2.4

                for i, (llm, alias) in enumerate(list(ALIAS_BY_MODEL_NAME.items())[::-1]):
                    run = get_textinjection_run(dataset, llm)
                    if run is None:
                        x_idx += 1.2
                        continue
                    x_vals = [x_idx]
                    y_vals = [run.summary[f'k{k}']]
                    plt.bar(x_vals, y_vals, 1, label=f"{alias} + G-RAG", color=COLOR_BY_MODEL_NAME[llm], hatch="//", alpha=.99)
                    x_idx += 1.2

                x_idx += 2.4

                for num_pseudo_words in PSEUDOWORDS:
                    run = get_sentenceformer_run(num_pseudo_words)
                    if run is None:
                        x_idx += 1
                        continue
                    x_vals = [x_idx]
                    y_vals = [run.summary.get(f'{dataset.lower()}_{testtrain}_k{k}', 0)]
                    color = COLOR_BY_CONCEPTFORMER[num_pseudo_words]
                    plt.bar(x_vals, y_vals, 1, label=f"GPT-2 0.1B + CF-{num_pseudo_words}", color=color, hatch="..", alpha=.99)
                    x_idx += 1.2

                plt.ylabel(f'Hits@{k}')

                plt.xticks([], [])

                plt.ylim(0, 1)
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
                plt.grid(color='grey', linestyle='-', alpha=0.5)

                plt.tight_layout()
                plt.savefig(os.path.join("output", f"7_final_results_{testtrain}_eval_hits_at_{k}_{dataset}.png"),
                            bbox_inches='tight')
                plt.savefig(os.path.join("output", f"7_final_results_{testtrain}_eval_hits_at_{k}_{dataset}.pdf"),
                            bbox_inches='tight')


if __name__ == "__main__":
    plot()
