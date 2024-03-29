from typing import List, Dict, Any
import os
import json

from tqdm import tqdm
from datasets import GeneratorBasedBuilder, SplitGenerator, Split, BuilderConfig, DatasetInfo
from datasets.features import Features, Value, Sequence


class TREx(GeneratorBasedBuilder):
    """
    A custom DatasetBuilder that loads the TREx dataset from JSON files.
    The dataset is divided into train (80%), validation (10%), and test (10%) splits.
    """
    VERSION = "1.0.0"
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="TREx",
            version=VERSION,
            description="A Large Scale Alignment of Natural Language with Knowledge Base Triples."
        )
    ]

    def _info(self) -> DatasetInfo:
        """
        Specifies the datasets.DatasetInfo object.
        """

        return DatasetInfo(
            features=Features({
                "docid": Value("string"),
                "title": Value("string"),
                "text": Value("string"),
                "uri": Value("string"),
                "words_boundaries": Sequence(Sequence(Value("int32"))),  # List of List of integers
                "sentences_boundaries": Sequence(Sequence(Value("int32"))),  # List of List of integers
                "triples": [{
                    "sentence_id": Value("int32"),
                    "subject": {
                        "boundaries": Sequence(Value("int32")),
                        "surfaceform": Value("string"),
                        "uri": Value("string"),
                        "annotator": Value("string")
                    },
                    "predicate": {
                        "boundaries": Sequence(Value("int32")),
                        "surfaceform": Value("string"),
                        "uri": Value("string"),
                        "annotator": Value("string")
                    },
                    "object": {
                        "boundaries": Sequence(Value("int32")),
                        "surfaceform": Value("string"),
                        "uri": Value("string"),
                        "annotator": Value("string")
                    },
                    "dependency_path": Value("string"),
                    "confidence": Value("float32"),
                    "annotator": Value("string")
                }],
                "entities": [{
                    "boundaries": Sequence(Value("int32")),
                    "surfaceform": Value("string"),
                    "uri": Value("string"),
                    "annotator": Value("string")
                }]
            })
        )

    def _split_generators(self, dl_manager: Any) -> List[SplitGenerator]:
        """
        Downloads the data and defines splits of the data.

        Args:
        - dl_manager: datasets.download.DownloadManager object

        Returns:
        - List of SplitGenerator objects for each data split.
        """
        urls = {"data": "https://figshare.com/ndownloader/files/8760241"}
        download_dir = dl_manager.download_and_extract(urls)

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "data_dir": download_dir["data"],
                    "split": "train",
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "data_dir": download_dir["data"],
                    "split": "validation",
                },
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "data_dir": download_dir["data"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, data_dir: str, split: str) -> Dict[str, Any]:
        """
        Generates examples for the dataset splits.

        Args:
        - data_dir: Path to the directory containing JSON data files.
        - split: Name of the split ('train', 'validation', 'test').

        Yields:
        - Examples for the given split.
        """
        json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        all_data_points = []

        # First, we accumulate all data points from all JSON files
        for filename in tqdm(json_files, desc="trex json files"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as file:
                data_list = json.load(file)
                all_data_points.extend(data_list)

        total_data_points = len(all_data_points)

        train_idx = int(0.8 * total_data_points)
        validation_idx = train_idx + int(0.1 * total_data_points)

        if split == "train":
            selected_data_points = all_data_points[:train_idx]
        elif split == "validation":
            selected_data_points = all_data_points[train_idx:validation_idx]
        elif split == "test":
            selected_data_points = all_data_points[validation_idx:]
        else:
            raise NotImplementedError(f"Split {split} not implemented.")

        for data in selected_data_points:
            yield data["uri"], data

