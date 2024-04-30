import glob
from typing import List, Dict, Any
from pathlib import Path

from datasets import GeneratorBasedBuilder, SplitGenerator, Split, BuilderConfig, DatasetInfo
from datasets.features import Features, Value


class TRExStarLite(GeneratorBasedBuilder):
    """
    A custom DatasetBuilder that loads the TREx dataset from JSON files.
    The dataset is divided into train (80%), validation (10%), and test (10%) splits.
    """
    VERSION = "1.0.0"
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="TRExStarLite",
            version=VERSION,
            description="TRExStar includes all relevant sub-graphs from the TREx Dataset. Each subgraph is a star topography with the relevant entity in the center and up to 100 neighbouring entities."
        )
    ]

    def _info(self) -> DatasetInfo:
        """
        Specifies the datasets.DatasetInfo object.
        """

        return DatasetInfo(
            features=Features({
                "entity": Value("string"),
                "json": Value("string"),
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
        trex_star_path = script_dir.parent.parent / 'data' / 'artifacts' / 'TRExStarLite_v1' / 'publish' / 'TRExStarLite_v1.tar'
        urls = {
            "trex_star": str(trex_star_path),
        }
        download_dir = dl_manager.download_and_extract(urls)

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "trex_star_dir": download_dir["trex_star"],
                    "split": "all",
                },
            ),
        ]

    def _generate_examples(self, trex_star_dir: str, split: str) -> Dict[str, Any]:
        """
        Generates examples for the dataset splits.

        Args:
        - graphs_dir: Path to the directory containing networkx graph files.
        - split: Name of the split ('train', 'validation', 'test').

        Yields:
        - Examples for the given split.
        """
        graph_files = glob.glob(f"{trex_star_dir}/*.json")
        if split != "all":
            raise NotImplementedError(f"Split {split} not implemented.")

        for graph_path in graph_files:
            entity_id = Path(graph_path).stem
            data = {}
            with open(graph_path, 'r') as f:
                graph_json = f.read()
            data["json"] = graph_json
            data["entity"] = entity_id
            yield entity_id, data
