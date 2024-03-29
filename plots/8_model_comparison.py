import os

import wandb
import matplotlib.pyplot as plt
import json

from plots.CONSTANTS import COLOR_BY_MODEL_NAME, LINESTYLE, ALIAS_BY_MODEL_NAME

api = wandb.Api()
PSEUDOWORDS = [1, 2, 3, 4, 5, 10, 15, 20]


def download_and_read_table(run, file_name):
    file_path = run.file(file_name).download(root=".", replace=True)  # Ensure it downloads to a specific path
    with open(file_path.name, 'r') as f:  # Use file_path.name to get the actual file name
        return json.load(f)

def get_baseline_run(dataset: str, llm: str):
    project = "university-of-zurich/6_eval_basemodel"
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
    return runs[0]

def plot():
    wandb.login()

    for dataset in ["TriREx", "TRExBite"]:

        for num_pseudo_words in PSEUDOWORDS:
            conceptformer_run = get_sentenceformer_run(num_pseudo_words)

            plt.figure(figsize=(10, 6), dpi=150)

            for file in conceptformer_run.files():
                if file.name.startswith(f"media/table/{dataset.lower()}_test_plot_table"):
                    # Download and read the JSON file
                    table_data = download_and_read_table(conceptformer_run, file.name)

                    # Extract x and y values for plotting
                    x_values, y_values = zip(*table_data["data"])

                    # Plot the data
                    plt.plot(x_values, y_values, label="GPT2-2 0.1B + CF", color="black",
                             linestyle=LINESTYLE["pseudowords"])

            for i, (llm, alias) in enumerate(list(ALIAS_BY_MODEL_NAME.items())[::-1]):
                run = get_baseline_run(dataset, llm)
                if run is None:
                    continue
                for file in run.files():
                    if file.name.startswith("media/table/plot_table"):
                        # Download and read the JSON file
                        table_data = download_and_read_table(run, file.name)

                        # Extract x and y values for plotting
                        x_values, y_values = zip(*table_data["data"])

                        color = COLOR_BY_MODEL_NAME[llm]

                        # Plot the data
                        plt.plot(x_values, y_values, label=alias, color=color, linestyle=LINESTYLE["base"])

            # Set axis labels
            plt.xlabel("k")
            plt.ylabel("hits@k")

            # Set axis limits
            plt.xlim(1, 50)
            plt.ylim(0, 1)

            # Add a legend
            plt.grid(color='grey', linestyle='-', alpha=0.5)
            plt.legend()

            plt.savefig(os.path.join("output", f"8_model_comparison_{dataset}_conceptformer-{num_pseudo_words}.png"), bbox_inches='tight')
            plt.savefig(os.path.join("output", f"8_model_comparison_{dataset}_conceptformer-{num_pseudo_words}.pdf"), bbox_inches='tight')
            plt.show()


if __name__ == "__main__":
    plot()
