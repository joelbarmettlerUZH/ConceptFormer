import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import umap
import colorsys

from tqdm.asyncio import tqdm

from src.Datasets.factory import trex_star_graphs_factory
from src.GraphAligner.BigGraphAligner import BigGraphAligner
from src.LLM.GPT2 import GPT2

script_directory = Path(__file__).parent.parent
data_directory = script_directory / "data"
output_directory = data_directory / f"output/ConceptFormer/"

plt.rcParams.update({'font.size': 16})

def plot():
    SEED = 1
    random.seed(SEED)
    device = torch.device("cuda:0")
    llm = GPT2(max_batch_size=1, device=device)

    for dataset_name in ["TRExStar"]:
        graphs = trex_star_graphs_factory(dataset_name)
        graph_aligner = BigGraphAligner(llm, graphs, dataset_name, use_untrained=True)

        take = 500_000
        selected_graphs = []
        for entity_id, G in graphs.items():
            if len(selected_graphs) == take:
                break
            if entity_id in graph_aligner.entity_index:
                selected_graphs.append(G)

        take = min(take, len(graphs))
        late_encoding_data = np.random.rand(take, llm.embedding_length)

        for i, G in tqdm(enumerate(selected_graphs), desc="Generating encodings"):
            entity_id = G.graph['central_node']
            late_encoding_data[i, :] = graph_aligner.node_embedding(entity_id).squeeze(0).squeeze(0).cpu().numpy()

        # Initialize UMAP. You can tweak these parameters.
        late_reducer = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine', transform_seed=SEED)

        # Fit the model and transform the data to 2D
        late_embedding = late_reducer.fit_transform(late_encoding_data)

        # Normalize the early embeddings to use for color mapping
        x = late_embedding[:, 0]
        y = late_embedding[:, 1]

        x_normalized = (x - np.min(x)) / (np.max(x) - np.min(x))
        y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y))

        # Convert to colors in HSL space and then to RGB
        colors = np.array([colorsys.hsv_to_rgb(xn, 0.8, 0.3 + 0.7 * yn) for xn, yn in zip(x_normalized, y_normalized)])

        # Determine the common axis limits
        x_min, x_max = np.min(late_embedding[:, 0]), np.max(late_embedding[:, 0])
        y_min, y_max = np.min(late_embedding[:, 1]), np.max(late_embedding[:, 1])

        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= 0.1 * x_range
        x_max += 0.1 * x_range
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range

        plt.figure(figsize=(10, 6), dpi=150)
        plt.scatter(late_embedding[:, 0], late_embedding[:, 1], color=colors, s=0.1)
        plt.xlim(x_min-5, x_max+5)
        plt.ylim(y_min-5, y_max+5)
        plt.title('UMAP Projection of Late Encoding Data', fontsize=14)
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.savefig(f"output/6_plot_embedding_analysis_late_embedding_umap_{dataset_name}.png", bbox_inches='tight')
        plt.show()
        plt.close()

        num_pseudo_words_list = [1, 2, 3, 4, 5, 10, 15, 20]

        for j, num_pseudo_words in enumerate(num_pseudo_words_list):
            publish_directory = output_directory / f"{num_pseudo_words}_context_vectors"

            graphgpt_encoding_list = []

            for graph_index, G in tqdm(enumerate(selected_graphs), desc="Generating encodings"):
                entity_id = G.graph['central_node']
                subject_file_path = publish_directory / f"{entity_id}.npy"

                # Check if the concept vector file exists
                if subject_file_path.exists():
                    # Load the concept vectors
                    try:
                        concept_vectors = torch.from_numpy(np.load(subject_file_path))
                    except ValueError:
                        continue

                    # Compute the mean of the concept vectors and add it to the list
                    mean_vector = concept_vectors.squeeze(0).mean(axis=0).to(device)
                    graphgpt_encoding_list.append(mean_vector.cpu().numpy())  # Convert to NumPy array and append

            # Convert the list of vectors into a 2D NumPy array
            graphgpt_encoding_data = np.array(graphgpt_encoding_list)

            graphgpt_reducer = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine', transform_seed=SEED)
            graphgpt_embedding = graphgpt_reducer.fit_transform(graphgpt_encoding_data)

            # Normalize the early embeddings to use for color mapping
            x = graphgpt_embedding[:, 0]
            y = graphgpt_embedding[:, 1]

            x_normalized = (x - np.min(x)) / (np.max(x) - np.min(x))
            y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y))

            # Convert to colors in HSL space and then to RGB
            colors = np.array(
                [colorsys.hsv_to_rgb(xn, 0.8, 0.3 + 0.7 * yn) for xn, yn in zip(x_normalized, y_normalized)])

            # Determine the common axis limits
            x_min, x_max = np.min(graphgpt_embedding[:, 0]), np.max(graphgpt_embedding[:, 0])
            y_min, y_max = np.min(graphgpt_embedding[:, 1]), np.max(graphgpt_embedding[:, 1])

            x_range = x_max - x_min
            y_range = y_max - y_min
            x_min -= 0.1 * x_range
            x_max += 0.1 * x_range
            y_min -= 0.1 * y_range
            y_max += 0.1 * y_range

            plt.figure(figsize=(10, 6), dpi=150)
            plt.scatter(graphgpt_embedding[:, 0], graphgpt_embedding[:, 1], color=colors, s=0.1)
            plt.xlim(x_min - 5, x_max + 5)
            plt.ylim(y_min - 5, y_max + 5)
            plt.title('UMAP Projection of Late Encoding Data', fontsize=14)
            plt.xlabel('UMAP Dimension 1')
            plt.ylabel('UMAP Dimension 2')
            plt.savefig(f"output/6_plot_embedding_analysis_conceptformer_{num_pseudo_words}_umap_{dataset_name}.png",
                        bbox_inches='tight')
            plt.show()
            plt.close()

if __name__ == "__main__":
    plot()
