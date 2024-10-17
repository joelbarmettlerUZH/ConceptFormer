import matplotlib.pyplot as plt
import numpy as np

from src.Datasets.TRExBite import TRExBite
from src.Datasets.TriREx import TriREx
from src.Datasets.TRExBiteLite import TRExBiteLite
from src.Datasets.TriRExLite import TriRExLite

plt.rcParams.update({'font.size': 16})

def plot_lite():
    trex_bite_lite_builder = TRExBiteLite()
    trirex_lite_builder = TriRExLite()


    trex_bite_lite_dataset_train = trex_bite_lite_builder.as_dataset(split='train')
    trex_bite_lite_dataset_val = trex_bite_lite_builder.as_dataset(split='validation')
    trex_bite_lite_dataset_test = trex_bite_lite_builder.as_dataset(split='test')

    trirex_lite_dataset_train = trirex_lite_builder.as_dataset(split='train')
    trirex_lite_dataset_val = trirex_lite_builder.as_dataset(split='validation')
    trirex_lite_dataset_test = trirex_lite_builder.as_dataset(split='test')


    # Data
    dataset_lite_train = [
        len(trex_bite_lite_dataset_train) // 1000,
        len(trirex_lite_dataset_train) // 1000,
    ]

    dataset_lite_val = [
        len(trex_bite_lite_dataset_val) // 1000,
        len(trirex_lite_dataset_val) // 1000,
    ]

    dataset_lite_test = [
        len(trex_bite_lite_dataset_test) // 1000,
        len(trirex_lite_dataset_test) // 1000,
    ]

    labels = [
        "T-Rex Bite Lite",
        "Tri-REx Lite",
    ]

    # Set the positions of the bars
    x = np.arange(len(labels))
    width = 0.25  # width of the bars

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)  # Double the width
    rects1 = ax.barh(x + width, dataset_lite_train, width, label='Train', color="#10a37f")
    rects2 = ax.barh(x, dataset_lite_val, width, label='Validation', color="#0b7258")
    rects3 = ax.barh(x - width, dataset_lite_test, width, label='Test', color="#064132")

    # Function to attach a label to each bar
    def autolabel(rects):
        for rect in rects:
            width = rect.get_width()
            ax.annotate('{:,.0f} k'.format(width),
                        xy=(width, rect.get_y() + rect.get_height() / 3),
                        xytext=(3, 0),  # 3 points horizontal offset
                        textcoords="offset points",
                        ha='left', va='center')

    # Add some text for labels, title and custom y-axis tick labels, etc.
    ax.set_xlabel('Datapoints (in thousands)')
    # ax.set_title('Sizes of datasets and splits')
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)

    # Attach labels to the bars
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # Adjust layout for better readability
    plt.tight_layout()

    # Show the plot
    plt.savefig(f"output/1_dataset_comparison_lite.png", bbox_inches='tight')
    plt.savefig(f"output/1_dataset_comparison_lite.pdf", bbox_inches='tight')
    plt.show()


def plot_full():
    trex_bite_builder = TRExBite()
    trirex_builder = TriREx()

    trex_bite_dataset_train = trex_bite_builder.as_dataset(split='train')
    trex_bite_dataset_val = trex_bite_builder.as_dataset(split='validation')
    trex_bite_dataset_test = trex_bite_builder.as_dataset(split='test')

    trirex_dataset_train = trirex_builder.as_dataset(split='train')
    trirex_dataset_val = trirex_builder.as_dataset(split='validation')
    trirex_dataset_test = trirex_builder.as_dataset(split='test')

    # Data
    dataset_train = [
        len(trex_bite_dataset_train) / 1_000_000,
        len(trirex_dataset_train) / 1_000_000,
    ]

    dataset_val = [
        len(trex_bite_dataset_val) / 1_000_000,
        len(trirex_dataset_val) / 1_000_000,
    ]

    dataset_test = [
        len(trex_bite_dataset_test) / 1_000_000,
        len(trirex_dataset_test) / 1_000_000,
    ]

    labels = [
        "T-Rex Bite",
        "Tri-REx",
    ]

    # Set the positions of the bars
    x = np.arange(len(labels))
    width = 0.25  # width of the bars

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)  # Double the width
    rects1 = ax.barh(x + width, dataset_train, width, label='Train', color="#10a37f")
    rects2 = ax.barh(x, dataset_val, width, label='Validation', color="#0b7258")
    rects3 = ax.barh(x - width, dataset_test, width, label='Test', color="#064132")

    # Function to attach a label to each bar
    def autolabel(rects):
        for rect in rects:
            width = rect.get_width()
            ax.annotate('{:,.2f} mio'.format(width),
                        xy=(width, rect.get_y() + rect.get_height() / 2),
                        xytext=(3, 0),  # 3 points horizontal offset
                        textcoords="offset points",
                        ha='left', va='center')

    # Add some text for labels, title and custom y-axis tick labels, etc.
    ax.set_xlabel('Datapoints (in millions)')
    # ax.set_title('Sizes of datasets and splits')
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)

    # Attach labels to the bars
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # Adjust layout for better readability
    plt.tight_layout()

    # Show the plot
    plt.savefig(f"output/1_dataset_comparison_full.png", bbox_inches='tight')
    plt.savefig(f"output/1_dataset_comparison_full.pdf", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_lite()
    plot_full()