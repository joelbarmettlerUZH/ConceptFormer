import json
from typing import Tuple, Dict

import networkx as nx
from datasets import Dataset
from tqdm import tqdm

from src.Datasets.WebQSPFinetuneSentences import WebQSPFinetuneSentences
from src.Datasets.WebQSPFinetuneStar import WebQSPFinetuneStar
from src.Datasets.WebQSPSentences import WebQSPSentences
from src.Datasets.WebQSPStar import WebQSPStar
from src.Datasets.TRExBite import TRExBite
from src.Datasets.TRExBiteLite import TRExBiteLite
from src.Datasets.TriREx import TriREx
from src.Datasets.TriRExLite import TriRExLite
from src.Datasets.TREx import TREx
from src.Datasets.TRExLite import TRExLite
from src.Datasets.TRExStar import TRExStar
from src.Datasets.TRExStarLite import TRExStarLite


def trex_factory(dataset_name: str) -> Tuple[Dataset, Dataset, Dataset]:
    if dataset_name == "TRExLite":
        trex_builder = TRExLite()
    elif dataset_name == "TREx":
        trex_builder = TREx()
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    if not trex_builder.info.splits:
        trex_builder.download_and_prepare()

    train_dataset = trex_builder.as_dataset(split="train")

    validation_dataset = trex_builder.as_dataset(split="validation")

    test_dataset = trex_builder.as_dataset(split="test")
    return train_dataset, validation_dataset, test_dataset

def trex_star_factory(dataset_name: str) -> Dataset:
    if dataset_name == "TRExStarLite":
        trex_star_builder = TRExStarLite()
    elif dataset_name == "TRExStar":
        trex_star_builder = TRExStar()
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    if not trex_star_builder.info.splits:
        trex_star_builder.download_and_prepare()

    return trex_star_builder.as_dataset(split="all")


def trex_bite_base_factory(dataset_name: str) -> Tuple[Dataset, Dataset, Dataset, Dict[str, nx.DiGraph]]:
    if dataset_name == "TRExBiteLite":
        trex_builder = TRExLite()
        trex_star_builder = TRExStarLite()
    elif dataset_name == "TRExBite":
        trex_builder = TREx()
        trex_star_builder = TRExStar()
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    if not trex_builder.info.splits:
        trex_builder.download_and_prepare()

    if not trex_star_builder.info.splits:
        trex_star_builder.download_and_prepare()

    graphs = {}
    for datapoint in tqdm(trex_star_builder.as_dataset(split="all"), desc="Loading nx graphs"):
        data = json.loads(datapoint['json'])
        graphs[datapoint['entity']] = nx.node_link_graph(data)

    train_dataset = trex_builder.as_dataset(split="train")
    validation_dataset = trex_builder.as_dataset(split="validation")
    test_dataset = trex_builder.as_dataset(split="test")

    return train_dataset, validation_dataset, test_dataset, graphs

def benchmark_base_factory(dataset_name: str) -> Dict[str, nx.DiGraph]:
    if dataset_name.endswith("Lite"):
        return trex_star_graphs_factory("TRExStarLite")
    return trex_star_graphs_factory("TRExStar")

def trex_star_graphs_factory(dataset_name: str)-> Dict[str, nx.DiGraph]:
    dataset = trex_star_factory(dataset_name)
    graphs = {}
    for datapoint in tqdm(dataset, desc="Loading nx graphs"):
        data = json.loads(datapoint['json'])
        graphs[datapoint['entity']] = nx.node_link_graph(data)
    return graphs


def trex_bite_factory(dataset_name: str) -> Tuple[Dataset, Dataset, Dataset]:
    if dataset_name == "TRExBiteLite":
        trex_bite_builder = TRExBiteLite()
    elif dataset_name == "TRExBite":
        trex_bite_builder = TRExBite()
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    if not trex_bite_builder.info.splits:
        trex_bite_builder.download_and_prepare()

    train_dataset = trex_bite_builder.as_dataset(split="train")
    validation_dataset = trex_bite_builder.as_dataset(split="validation")
    test_dataset = trex_bite_builder.as_dataset(split="test")
    return train_dataset, validation_dataset, test_dataset

def trirex_factory(dataset_name: str) -> Tuple[Dataset, Dataset, Dataset]:
    if dataset_name == "TriRExLite":
        trirex_builder = TriRExLite()
    elif dataset_name == "TriREx":
        trirex_builder = TriREx()
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    if not trirex_builder.info.splits:
        trirex_builder.download_and_prepare()

    train_dataset = trirex_builder.as_dataset(split="train")
    validation_dataset = trirex_builder.as_dataset(split="validation")
    test_dataset = trirex_builder.as_dataset(split="test")

    return train_dataset, validation_dataset, test_dataset

def rex_factory(dataset_name: str) -> Tuple[Dataset, Dataset, Dataset, Dict[str, nx.DiGraph]]:
    if dataset_name == "TRExBiteLite":
        dataset_builder = TRExBiteLite()
        trex_star_builder = TRExStarLite()
    elif dataset_name == "TRExBite":
        dataset_builder = TRExBite()
        trex_star_builder = TRExStar()
    elif dataset_name == "TriRExLite":
        dataset_builder = TriRExLite()
        trex_star_builder = TRExStarLite()
    elif dataset_name == "TriREx":
        dataset_builder = TriREx()
        trex_star_builder = TRExStar()
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    if not dataset_builder.info.splits:
        dataset_builder.download_and_prepare()

    if not trex_star_builder.info.splits:
        trex_star_builder.download_and_prepare()

    train_dataset = dataset_builder.as_dataset(split="train")
    validation_dataset = dataset_builder.as_dataset(split="validation")
    test_dataset = dataset_builder.as_dataset(split="test")

    graphs = {}
    for datapoint in tqdm(trex_star_builder.as_dataset(split="all"), desc="Loading nx graphs"):
        data = json.loads(datapoint['json'])
        graphs[datapoint['entity']] = nx.node_link_graph(data)

    return train_dataset, validation_dataset, test_dataset, graphs

def rex_raw_factory(dataset_name: str) -> Tuple[Dataset, Dataset, Dataset]:
    if dataset_name == "TRExBiteLite":
        dataset_builder = TRExBiteLite()
    elif dataset_name == "TRExBite":
        dataset_builder = TRExBite()
    elif dataset_name == "TriRExLite":
        dataset_builder = TriRExLite()
    elif dataset_name == "TriREx":
        dataset_builder = TriREx()
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    if not dataset_builder.info.splits:
        dataset_builder.download_and_prepare()

    train_dataset = dataset_builder.as_dataset(split="train")
    validation_dataset = dataset_builder.as_dataset(split="validation")
    test_dataset = dataset_builder.as_dataset(split="test")

    return train_dataset, validation_dataset, test_dataset

def web_qsp_factory() -> Tuple[Dataset, Dict[str, nx.DiGraph]]:
    web_qsp_sentence_builder = WebQSPSentences()
    web_qsp_star_builder = WebQSPStar()

    if not web_qsp_sentence_builder.info.splits:
        web_qsp_sentence_builder.download_and_prepare()

    if not web_qsp_star_builder.info.splits:
        web_qsp_star_builder.download_and_prepare()

    test_dataset = web_qsp_sentence_builder.as_dataset(split="test")

    graphs = {}
    for datapoint in tqdm(web_qsp_star_builder.as_dataset(split="all"), desc="Loading nx graphs"):
        data = json.loads(datapoint['json'])
        graphs[datapoint['entity']] = nx.node_link_graph(data)

    return test_dataset, graphs

def web_qsp_finetune_factory() -> Tuple[Dataset, Dataset, Dict[str, nx.DiGraph]]:
    web_qsp_sentence_builder = WebQSPFinetuneSentences()
    web_qsp_star_builder = WebQSPFinetuneStar()

    if not web_qsp_sentence_builder.info.splits:
        web_qsp_sentence_builder.download_and_prepare()

    if not web_qsp_star_builder.info.splits:
        web_qsp_star_builder.download_and_prepare()

    train_dataset = web_qsp_sentence_builder.as_dataset(split="train")
    test_dataset = web_qsp_sentence_builder.as_dataset(split="test")

    graphs = {}
    for datapoint in tqdm(web_qsp_star_builder.as_dataset(split="all"), desc="Loading nx graphs"):
        data = json.loads(datapoint['json'])
        graphs[datapoint['entity']] = nx.node_link_graph(data)

    return train_dataset, test_dataset, graphs