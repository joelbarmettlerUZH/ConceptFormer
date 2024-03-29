import csv
import glob
from typing import List, Dict, Any
from pathlib import Path

from datasets import GeneratorBasedBuilder, SplitGenerator, Split, BuilderConfig, DatasetInfo
from datasets.features import Features, Value, Sequence


class WebQSPSentences(GeneratorBasedBuilder):
    VERSION = "1.0.1"
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="WebQSPSentences",
            version=VERSION,
            description=("")
        )
    ]

    def _info(self) -> DatasetInfo:
        """
        Specifies the datasets.DatasetInfo object.
        """

        return DatasetInfo(
            features=Features({
                "sentence": Value("string"),
                "k": Value("int32"),
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
        web_qsp_path = script_dir.parent.parent / 'data' / 'artifacts' / 'WebQSPSentences_v1' / 'publish' / 'WebQSPSentences_v1.tar'
        urls = {
            "web_qsp_sentences_dir": str(web_qsp_path),
        }
        download_dir = dl_manager.download_and_extract(urls)

        return [
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "web_qsp_sentences_dir": download_dir["web_qsp_sentences_dir"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, web_qsp_sentences_dir: str, split: str) -> Dict[str, Any]:
        selected_data_points = glob.glob(f"{web_qsp_sentences_dir}/{split}/*.csv")

        for csv_file in selected_data_points:
            question_id = Path(csv_file).stem
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)

                # Yielding each row
                for idx, row in enumerate(reader):
                    datapoint = {
                        "sentence": row['sentence'],
                        "k": row['k'],
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
                    yield f'{question_id}-{idx}', datapoint
