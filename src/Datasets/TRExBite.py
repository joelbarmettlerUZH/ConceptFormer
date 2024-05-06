import csv
import glob
from typing import List, Dict, Any
from pathlib import Path

from datasets import GeneratorBasedBuilder, SplitGenerator, Split, BuilderConfig, DatasetInfo
from datasets.features import Features, Value, Sequence


class TRExBite(GeneratorBasedBuilder):
    VERSION = "1.0.0"
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="TRExBite",
            version=VERSION,
            description=("TRExBite is generated from the TREx-Dataset, extracting individual sentences with a "
                        "maximum length of 512 in which a subject is guaranteed to be located before its "
                        "corresponding object, and the object is guaranteed to be at the end of the sentence.")
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
                    "rank": Value("string"),
                    "boundaries": Sequence(Value("int32"))  # List of integers
                },
                "predicate": {
                    "id": Value("string"),
                    "label": Value("string")
                },
                "object": {
                    "id": Value("string"),
                    "label": Value("string"),
                    "rank": Value("string"),
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
        trex_bite_path = script_dir.parent.parent / 'data' / 'artifacts' / 'TRExBite_v1' / 'publish' / 'TRExBite_v1.tar'
        urls = {
            "trex_bite": str(trex_bite_path),
        }
        download_dir = dl_manager.download_and_extract(urls)

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "trex_bite_dir": download_dir["trex_bite"],
                    "split": "train",
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "trex_bite_dir": download_dir["trex_bite"],
                    "split": "validation",
                },
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "trex_bite_dir": download_dir["trex_bite"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, trex_bite_dir: str, split: str) -> Dict[str, Any]:
        selected_data_points = glob.glob(f"{trex_bite_dir}/{split}/*.csv")

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
