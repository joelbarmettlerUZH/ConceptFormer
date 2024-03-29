import glob
from typing import List, Dict, Any
from pathlib import Path

from datasets import GeneratorBasedBuilder, SplitGenerator, Split, BuilderConfig, DatasetInfo
from datasets.features import Features, Value


class WebQSPStar(GeneratorBasedBuilder):
    VERSION = "1.0.6"
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="WebQSPStar",
            version=VERSION,
            description=""
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
        trex_star_path = script_dir.parent.parent / 'data' / 'artifacts' / 'WebQSPStar_v1' / 'publish' / 'WebQSPStar_v1.tar'
        urls = {
            "web_qsp_star": str(trex_star_path),
        }
        download_dir = dl_manager.download_and_extract(urls)

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "web_qsp_star": download_dir["web_qsp_star"],
                    "split": "all",
                },
            ),
        ]

    def _generate_examples(self, web_qsp_star: str, split: str) -> Dict[str, Any]:
        """
        Generates examples for the dataset splits.

        Args:
        - graphs_dir: Path to the directory containing networkx graph files.
        - split: Name of the split ('train', 'validation', 'test').

        Yields:
        - Examples for the given split.
        """
        graph_files = glob.glob(f"{web_qsp_star}/*.json")
        print(web_qsp_star)
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
