import argparse

from src.Datasets.factory import trex_factory

# Set up argument parser
parser = argparse.ArgumentParser(description='Process dataset name.')
parser.add_argument('--dataset_name', type=str, default='TREx',
                    help='Name of the dataset to use')

# Parse arguments
args = parser.parse_args()
DATASET_NAME = args.dataset_name

if __name__== "__main__":
    train_dataset, validation_dataset, test_dataset = trex_factory(DATASET_NAME)
    print(f"{DATASET_NAME}:train", len(train_dataset), train_dataset[0])
    print(f"{DATASET_NAME}:validation", len(validation_dataset), validation_dataset[0])
    print(f"{DATASET_NAME}:test", len(test_dataset), test_dataset[0])
