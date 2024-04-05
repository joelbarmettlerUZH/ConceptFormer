from pathlib import Path
from typing import Optional, List

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, LlamaTokenizer, LlamaForCausalLM

from .LLM import LLM


class Llama2(LLM):
    """
    A class that wraps the Llama2 model for embedding computations.
    """

    def __init__(self, model_name_or_path: str | Path, max_batch_size=8, device=torch.device('cuda'), bits=2):
        """
        Initialize the Llama2 wrapper with a specific model path.

        Args:
            model_path (str): The path to the pretrained Llama2 model.
        """
        context_length = 2048
        llm_name = model_name_or_path.replace("/", "-")
        if isinstance(model_name_or_path, Path):
            tokenizer = LlamaTokenizer.from_pretrained(str(model_name_or_path), legacy=False, clean_up_tokenization_spaces=True)
            pretrained = LlamaForCausalLM.from_pretrained(str(model_name_or_path))
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, legacy=False, clean_up_tokenization_spaces=True)

            is_quantized = "gptq" in model_name_or_path.lower()
            gptq_config = None
            if is_quantized:
                llm_name += f"-{bits}bit"
                gptq_config = GPTQConfig(bits=bits, disable_exllama=True, tokenizer=tokenizer,
                                         max_input_length=context_length, use_cuda_fp16=True)
            pretrained = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                quantization_config=gptq_config,
                trust_remote_code=True,
                revision="main",
            )

        super().__init__(
            model=pretrained,
            tokenizer=tokenizer,
            embedding_length=pretrained.config.hidden_size,
            max_batch_size=max_batch_size,
            max_sequence_length=context_length,
            space_token="▁",
            name=llm_name,
            device=device,
            model_name_or_path=model_name_or_path,
        )
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._bos_token_embedding = self.token_ids_to_embeddings(torch.Tensor([[1]]).to(torch.int).to(self._device))

    def add_positional_encoding(self, encodings: torch.Tensor) -> torch.Tensor:
        return encodings

    def add_special_token_embeddings(self, encodings: torch.Tensor) -> torch.Tensor:
        assert encodings.ndim == 3, "Input must be batched"
        assert encodings.shape[2] == self.embedding_length, "Input does not match embedding size"

        batch_size = encodings.shape[0]
        repeated_bos_tensor = self._bos_token_embedding.repeat(batch_size, 1, 1)
        concatenated_tensor = torch.cat((repeated_bos_tensor, encodings), dim=1)
        return concatenated_tensor[:, :self.max_sequence_length, :]

    def token_ids_to_embeddings(self, input_ids):
        """
        Converts token ids to model input embeddings
        :param input_ids: Torch tensor of shape (batch_size, n_tokens)
        :return: Tensor of size (batch_size, n_tokens, sequence_length)
        """
        assert input_ids.ndim == 2, f"Input must be a tensor of size (batch, n_tokens)"
        assert input_ids.dtype in (
            torch.int, torch.int8, torch.int16, torch.int32, torch.int64), "The tensor is not of integer type"

        with torch.no_grad():
            embeddings = self._model.base_model.embed_tokens(input_ids)

        assert embeddings.shape[0] == input_ids.shape[0], "Batch size changed unexpectedly"
        assert embeddings.shape[1] == input_ids.shape[1], "Number of tokens changed unexpectedly"
        assert embeddings.shape[2] == self.embedding_length, "Wrong embedding length"

        return embeddings.to(self._device)

    def minimal_batch_input_ids_to_early_embeddings(self, batch_input_ids: torch.Tensor) -> torch.Tensor:
        return self._model.base_model.embed_tokens(batch_input_ids)

    def minimal_batch_input_ids_to_late_embeddings(self, batch_input_ids: torch.Tensor,
                                                   batch_attention_masks: Optional[
                                                       torch.Tensor] = None) -> torch.Tensor:
        outputs = self._model.base_model(batch_input_ids, attention_mask=batch_attention_masks)
        return outputs.last_hidden_state

    def attention_mask(self, batch_attention_mask: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(batch_attention_mask, dim=0)

    def evaluate(self, data: DataLoader, criterion=MSELoss, batch_size: int = 8,
                 max_length: int = 4096) -> float:
        """
        Evaluate the performance of the model using a given dataset and loss criterion.
        """
        self._model.eval()

        # Instantiate the criterion
        loss_function = criterion()
        total_loss = 0.0
        total_steps = 0

        # Chunk loading
        for chunk in data:
            # Tokenize the chunked data
            tokenized_data = self._tokenizer(chunk['text'], truncation=True, padding='max_length',
                                             max_length=max_length, return_attention_mask=True, return_tensors="pt")
            tokenized_loader = DataLoader(tokenized_data, batch_size=batch_size, shuffle=True)

            with torch.no_grad():
                for batch in tqdm(tokenized_loader, desc="Evaluating"):
                    # Move the tokenized data to GPU
                    input_ids = batch['input_ids'].to(self._device)
                    attention_mask = batch['attention_mask'].to(self._device)

                    # Obtain the model's outputs
                    outputs = self._model.transformer(input_ids, attention_mask=attention_mask).last_hidden_state

                    # Get the embeddings of the input IDs as targets
                    target_embeddings = self._model.transformer.wte(input_ids)

                    # Calculate the loss between the model's outputs and the original token embeddings
                    loss = loss_function(outputs[:, :-1], target_embeddings[:, 1:])

                    total_loss += loss.item()
                    total_steps += 1

        # Ensure we have processed some steps
        if total_steps == 0:
            raise ValueError("No data was processed during evaluation. Check your data format and content.")

        return total_loss / total_steps
