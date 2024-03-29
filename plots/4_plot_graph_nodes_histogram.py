import os
import matplotlib.pyplot as plt
import numpy as np

from src.Datasets.factory import trex_star_graphs_factory

plt.rcParams.update({'font.size': 16})

def plot():
    graphs = trex_star_graphs_factory('TRExStar')

    node_counts = []
    for graph in graphs.values():
        node_counts.append(graph.number_of_nodes() - 1)

    plt.figure(figsize=(10, 6), dpi=150)
    counts, bins, patches = plt.hist(node_counts, bins=10, edgecolor='black', color="#0b7258")

    # plt.title(f"Subgraph sizes in T-Rex Star")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Frequency")
    plt.xlim(1, 101)
    plt.grid(False)
    plt.tight_layout()
    plt.yscale('log')

    bin_centers = 0.5 * np.diff(bins) + bins[:-1]
    custom_labels = [f"{i}-{i+9}" for i in range(1, 100, 10)]
    plt.xticks(bin_centers, custom_labels)


    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir, f"4_graph_nodes_histogram.png"), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"4_graph_nodes_histogram.pdf"), bbox_inches='tight')
    plt.show()




if __name__ == "__main__":
    plot()
