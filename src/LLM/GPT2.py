from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from torch.nn import MSELoss, Module
from typing import Type, Optional, List
from torch.utils.data import DataLoader
from .LLM import LLM


class GPT2(LLM):
    """
    A class that wraps the GPT-2 model for embedding computations.
    """

    def __init__(self, model_name_or_path='gpt2', max_batch_size=8, device=torch.device('cuda')):
        pretrained = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        super().__init__(
            model=pretrained,
            tokenizer=GPT2TokenizerFast.from_pretrained(model_name_or_path),
            embedding_length=pretrained.config.n_embd,
            max_batch_size=max_batch_size,
            max_sequence_length=1024,
            space_token="Ġ",
            name=model_name_or_path.replace("/", "-"),
            device=device,
            model_name_or_path=model_name_or_path
        )
        self._tokenizer.add_special_tokens({'pad_token': self._tokenizer.eos_token})
        # self._model.resize_token_embeddings(len(self._tokenizer), pad_to_multiple_of=1)

    def add_positional_encoding(self, encodings: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encodings to the embeddings if the model needs that
        :param encodings: input encodings
        :return: torch.tensor of same shape as input
        """
        assert encodings.ndim == 3, f"Encodings must be of shape (batch_size, sequence_length, {self.embedding_length}), got {encodings.shape}"
        assert encodings.shape[
                   2] == self.embedding_length, f"Embedding shape mismatch. Model requires {self.embedding_length}, got {encodings.shape[2]}"

        seq_length = encodings.size(1)
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=self._device).unsqueeze(0)
        positional_embeddings = self._model.transformer.wpe(position_ids)
        input_embeddings_with_position = encodings + positional_embeddings

        assert input_embeddings_with_position.shape == encodings.shape, "Unexpected shape change"
        return input_embeddings_with_position

    def add_special_token_embeddings(self, encodings: torch.Tensor) -> torch.Tensor:
        """
        GPT-2 has no special tokens, hence there are no embeddings

        :param encodings:
        :return:
        """
        return encodings

    def token_ids_to_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Converts token ids to model input embeddings
        :param input_ids: Torch tensor of shape (batch_size, n_tokens)
        :return: Tensor of size (batch_size, n_tokens, sequence_length)
        """
        assert input_ids.ndim == 2, f"Input must be a tensor of size (batch, n_tokens)"
        assert input_ids.dtype in (
            torch.int, torch.int8, torch.int16, torch.int32, torch.int64), "The tensor is not of integer type"

        with torch.no_grad():
            embeddings = self._model.transformer.wte(input_ids)

        assert embeddings.shape[0] == input_ids.shape[0], "Batch size changed unexpectedly"
        assert embeddings.shape[1] == input_ids.shape[1], "Number of tokens changed unexpectedly"
        assert embeddings.shape[2] == self.embedding_length, "Wrong embedding length"

        return embeddings.to(self._device)

    def minimal_batch_input_ids_to_early_embeddings(self, batch_input_ids: torch.Tensor) -> torch.Tensor:
        return self._model.transformer.wte(batch_input_ids)

    def minimal_batch_input_ids_to_late_embeddings(self, batch_input_ids: torch.Tensor, batch_attention_masks: Optional[
        torch.Tensor] = None) -> torch.Tensor:
        output = self._model.transformer(batch_input_ids, attention_mask=batch_attention_masks)
        return output.last_hidden_state

    def attention_mask(self, batch_attention_mask: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(batch_attention_mask, dim=0)

    def evaluate(self, data: DataLoader, criterion: Type[Module] = MSELoss, batch_size: int = 8,
                 max_length: int = 1024) -> float:
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
