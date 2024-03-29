from typing import TYPE_CHECKING
from collections import defaultdict

from torch.utils.data import Sampler
import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from src.DataLoaders.RExEmbeddingDynamicLoader import RExEmbeddingDynamicLoader


class DynamicNeighbourBatchSampler(Sampler):
    def __init__(self, dataset: "RExEmbeddingDynamicLoader", batch_size: int, shuffle: bool, drop_last: bool):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indices = list(range(len(dataset)))
        self.dimension_to_indices = defaultdict(list)
        self._group_indices_by_dimension()

    def _group_indices_by_dimension(self):
        self.dimension_to_indices.clear()
        for idx in tqdm(self.indices, desc="Grouping batches"):
            number_of_neighbours = self.dataset.num_neighbours_at_index(idx)
            self.dimension_to_indices[number_of_neighbours].append(idx)

    def _create_batches(self):
        batches = []
        for dimension, indices in tqdm(self.dimension_to_indices.items(), desc="Creating batches"):
            if self.shuffle:
                np.random.shuffle(indices)
            batches += [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return batches

    def __iter__(self):
        if self.shuffle:
            self._group_indices_by_dimension()
        self.batches = self._create_batches()
        for batch in self.batches:
            yield batch

    def __len__(self):
        total_batches = 0
        for _, indices in self.dimension_to_indices.items():
            if self.shuffle:
                # If shuffling, the number of indices might not change, so we can calculate directly
                total_batches += len(indices) // self.batch_size
            else:
                # If not shuffling, directly use the division to count full batches
                batch_count = len(indices) // self.batch_size
                if not self.drop_last:
                    # If not dropping the last batch, add one to count for the remaining items (if any)
                    batch_count += int(len(indices) % self.batch_size > 0)
                total_batches += batch_count
        return total_batches

