import glob
from typing import List, Dict, Any
from pathlib import Path
import csv

from datasets import GeneratorBasedBuilder, SplitGenerator, Split, BuilderConfig, DatasetInfo
from datasets.features import Features, Value, Sequence


class TriREx(GeneratorBasedBuilder):
    """
    A custom DatasetBuilder that loads the TREx dataset from JSON files.
    The dataset is divided into train (80%), validation (10%), and test (10%) splits.
    """
    VERSION = "1.0.1"
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="TriREx",
            version=VERSION,
            description="TriREx is a large scale dataset, generated using the Mistral-7B language model, consisting of short sentences in a semi-flexible subject-predicate-object format."
        )
    ]

    def _info(self) -> DatasetInfo:
        """
        Specifies the datasets.DatasetInfo object.
        """

        return DatasetInfo(
            features=Features({
                "sentence": Value("string"),
                "subject": {
                    "id": Value("string"),
                    "label": Value("string"),
                    "rank": Value("float32"),
                    "boundaries": Sequence(Value("int32"))  # List of integers
                },
                "predicate": {
                    "id": Value("string"),
                    "label": Value("string")
                },
                "object": {
                    "id": Value("string"),
                    "label": Value("string"),
                    "rank": Value("float32"),
                    "boundaries": Sequence(Value("int32"))  # List of integers
                }
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
        script_dir = Path(__file__).parent
        trirex_path = script_dir.parent.parent / 'data' / 'artifacts' / 'TriREx_v1' / 'publish' / 'TriREx_v1.tar'
        urls = {
            "trirex": str(trirex_path),
        }
        download_dir = dl_manager.download_and_extract(urls)

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "trirex_dir": download_dir["trirex"],
                    "split": "train",
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "trirex_dir": download_dir["trirex"],
                    "split": "validation",
                },
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "trirex_dir": download_dir["trirex"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, trirex_dir: str, split: str) -> Dict[str, Any]:
        selected_data_points = glob.glob(f"{trirex_dir}/{split}/*.csv")

        for csv_file in selected_data_points:
            entity_id = Path(csv_file).stem
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)

                # Yielding each row
                for idx, row in enumerate(reader):
                    datapoint = {
                        "sentence": row['sentence'],
                        "subject": {
                            "id": row['subject_id'],
                            "label": row['subject_label'],
                            "rank": float(row['subject_rank']),
                            "boundaries": [
                                int(row['subject_boundary_start']),
                                int(row['subject_boundary_end']),
                            ]
                        },
                        "predicate": {
                            "id": row['predicate_id'],
                            "label": row['predicate_label'],
                        },
                        "object": {
                            "id": row['object_id'],
                            "label": row['object_label'],
                            "rank": float(row['object_rank']),
                            "boundaries": [
                                int(row['object_boundary_start']),
                                int(row['object_boundary_end']),
                            ]
                        }
                    }
                    yield f'{entity_id}-{idx}', datapoint