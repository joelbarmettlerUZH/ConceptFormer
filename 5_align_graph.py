import argparse

import torch

from src.Datasets.TRExStar import TRExStar
from src.Datasets.TRExStarLite import TRExStarLite
from src.Datasets.factory import trex_star_graphs_factory
from src.GraphAligner.BigGraphAligner import BigGraphAligner
from src.LLM.GPT2 import GPT2
from src.LLM.Llama2 import Llama2

parser = argparse.ArgumentParser(description='Process dataset parameters.')
parser.add_argument('--dataset_name', type=str, default='TRExStarLite',
                    help='Name of the dataset to generate (TRExStar or TRExStarLite)')
parser.add_argument('--llm_type', type=str, default='llama-2',
                    help='Type of Large Language Model to generate initial encodings (gpt-2 or llama-2)')
parser.add_argument('--llm_name_or_path', type=str, default='TheBloke/Llama-2-7B-GPTQ',
                    help='Name of huggingface model (gpt2, gpt2-medium, gpt2-large, gpt2-xl, openlm-research/open_llama_3b_v2, TheBloke/Llama-2-7B-GPTQ)')
parser.add_argument('--gpu', type=int, default=-1,
                    help='Index of GPU to use')

args = parser.parse_args()
DATASET_NAME = args.dataset_name
LLM_TYPE = args.llm_type
LLM_PATH_OR_NAME = args.llm_name_or_path
GPU = args.gpu

if __name__ == "__main__":
    device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() and GPU > -1 else "cpu")
    if LLM_TYPE == "gpt-2":
        llm = GPT2(model_name_or_path=LLM_PATH_OR_NAME, max_batch_size=1, device=device)
    elif LLM_TYPE == "llama-2":
        llm = Llama2(model_name_or_path=LLM_PATH_OR_NAME, max_batch_size=1, device=device)
    else:
        raise ValueError(f"Unknown model {LLM_TYPE}")

    if DATASET_NAME == "TRExStarLite":
        builder = TRExStarLite()
    elif DATASET_NAME == "TRExStar":
        builder = TRExStar()
    else:
        raise ValueError(f"Unknown dataset {DATASET_NAME}")

    if not builder.info.splits:
        builder.download_and_prepare()

    dataset = builder.as_dataset(split='all')

    graphs = trex_star_graphs_factory(DATASET_NAME)
    graph_aligner = BigGraphAligner(llm, graphs, DATASET_NAME)

    print(f"Entities: {len(graph_aligner.entity_index)}")
    print(f"Relations: {len(graph_aligner.relation_index)}")
