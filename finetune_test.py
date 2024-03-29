from pathlib import Path

import torch

from src.LLM.factory import llm_factory

script_directory = Path(__file__).parent
data_directory = script_directory / "data"
artifacts_finetune_directory = data_directory / f"artifacts/WebQSPFineTuneGPT2"

device = torch.device(f"cuda:0")
llm = llm_factory(
    "gpt-2",
    embedding_llm_name=str(artifacts_finetune_directory),
    batch_size=1,
    device=device,
    bits=1
)
question = "What is the capital of France?"
sentence = "Question: " + question + "\nAnswer:"
embeddings = llm.early_embedding(sentence)
output = llm._model(inputs_embeds=embeddings)
last_logits = output.logits[:, -1, :]
top_logits, top_indices = torch.topk(last_logits, 1, dim=-1)
selected_indices = torch.tensor([indices[0] for indices in top_indices]).to(llm._device)
token_str = llm._tokenizer.convert_ids_to_tokens([selected_indices[0].item()])[0]
print(token_str)


