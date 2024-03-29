import torch

from src.LLM.GPT2 import GPT2
from src.LLM.LLM import LLM
from src.LLM.Llama2 import Llama2


def llm_factory(llm_type: str, embedding_llm_name: str, batch_size: int, device: torch.device, bits: int) -> LLM:
    if llm_type == "gpt-2":
        return GPT2(model_name_or_path=embedding_llm_name, max_batch_size=batch_size, device=device)
    elif llm_type == "llama-2":
        return Llama2(model_name_or_path=embedding_llm_name, max_batch_size=batch_size, device=device, bits=bits)
    else:
        raise ValueError(f"Unknown model type {llm_type}")
