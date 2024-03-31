import logging
import random
from collections import OrderedDict
from typing import List, Optional, Dict

from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


class LLM(ABC):
    """
    A wrapper class for language models like GPT-2 from huggingface.
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, name: str,
                 embedding_length: int, model_name_or_path: str, cache_size: int = 5_000_000,
                 max_batch_size=8, max_sequence_length=1024, space_token=' ', device=torch.device('cuda')) -> None:
        self.name = name
        self.model_name_or_path = model_name_or_path
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.space_token = space_token
        self._device = device
        self._model = model.to(self._device)
        self._tokenizer = tokenizer
        self._embedding_length = embedding_length
        self._cache_size = cache_size
        self._embeddings_cache = OrderedDict()

        self.pooling_strategy = torch.mean

        for param in self._model.parameters():
            param.requires_grad = False

    @abstractmethod
    def add_positional_encoding(self, encodings: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def add_special_token_embeddings(self, encodings: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def token_ids_to_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def evaluate(self, data: DataLoader, criterion, batch_size, max_length) -> float:
        pass

    @abstractmethod
    def minimal_batch_input_ids_to_early_embeddings(self, batch_input_ids: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def minimal_batch_input_ids_to_late_embeddings(self, batch_input_ids: torch.Tensor, batch_attention_masks: Optional[
        torch.Tensor] = None) -> torch.Tensor:
        pass

    @abstractmethod
    def attention_mask(self, batch_attention_mask: List[torch.Tensor]) -> torch.Tensor:
        pass

    def _clear_cache(self):
        """
        Clears the late embeddings cache
        :return:
        """
        self._embeddings_cache.clear()

    def _cache_late_embedding(self, text: str, embedding: torch.Tensor) -> None:
        """
        Caches late embeddings
        :param text: Text to cache
        :param embedding: Tensor(1, 1, embedding_dim)
        :return:
        """
        if text in self._embeddings_cache:
            del self._embeddings_cache[text]
        elif len(self._embeddings_cache) >= (self._cache_size):
            self._embeddings_cache.popitem(last=False)
        self._embeddings_cache[text] = embedding.to('cpu')

    def _retrieve_late_embedding(self, text: str) -> torch.Tensor | None:
        """
        Retrieves embedding from late embeddings cache, or returns None
        :param text: text to retrieve
        :return:
        """
        cached = self._embeddings_cache.get(text, None)
        if cached is None:
            return
        return cached.to(self._device)

    def text_to_input_ids(self, text: str) -> torch.Tensor:
        """
        Converts a text to its token ids.
        :param text: Text to tokenize
        :return: Tensor of shape (1, n_tokens)
        """
        assert text and len(text) > 0, "Text cannot be empty"

        token_ids = self._tokenizer.encode(text, return_tensors="pt", max_length=self.max_sequence_length,
                                           truncation=True,
                                           add_special_tokens=False).to(self._device)

        assert token_ids.shape[0] == 1, "Text embedding is not auto-batched with batch size 1"
        return token_ids

    def early_embedding(self, text: str) -> torch.Tensor:
        """
        Generates text embeddings using the word to encodings matrix only.

        :param text: Text to encode
        :return: Tensor of size (1, n_tokens, embedding_dim)
        """
        assert text and len(text) > 0, "Text cannot be empty"

        embedding = self.early_embedding_batch([text])
        n_tokens = self.text_to_input_ids(text).shape[1]

        if self._tokenizer.padding_side == 'right':
            embedding = embedding[:, :n_tokens, :]
        else:
            embedding = embedding[:, -n_tokens:, :]

        assert embedding.shape == (1, n_tokens, self.embedding_length), "Early embedding shape mismatch"
        return embedding

    def late_embedding(self, text: str) -> torch.Tensor:
        """
        Generates text embeddings by extracting the hidden state from the last possible transformer block
        As late embeddings are always pooled, you get 1 vector per text
        :param text: Text to encode
        :return: Tensor of size (1, 1, embedding_dim)
        """
        assert text and len(text) > 0, "Text cannot be empty"

        embedding = self._retrieve_late_embedding(text)
        if embedding is not None:
            return embedding

        tokenized_texts = self._tokenizer.batch_encode_plus(
            [text], return_tensors="pt", max_length=self.max_sequence_length, truncation=True,
            add_special_tokens=False
        )

        input_ids = tokenized_texts["input_ids"].to(self._device)
        encodings = self.minimal_batch_input_ids_to_late_embeddings(input_ids)
        embedding = self.pooling_strategy(encodings, dim=1).unsqueeze(1)

        self._cache_late_embedding(text, embedding)

        assert embedding.shape == (1, 1, self.embedding_length), "Late embedding shape mismatch"
        return embedding

    def early_embedding_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Performs early embedding but for a whole batch efficiently.
        All embeddings are padded to a dimension of 1024

        :param texts: Texts to encode
        :return: Tensor of size (batch_size, 1024, embedding_dim)
        """
        assert len(texts) > 0, "No texts to encode"
        assert all(text and len(text) > 0 for text in texts), "Texts can not be empty"

        embeddings = []

        tokenized_texts = self._tokenizer.batch_encode_plus(
            texts, return_tensors="pt", max_length=self.max_sequence_length, truncation=True,
            padding='max_length', add_special_tokens=False
        )

        input_ids = tokenized_texts["input_ids"].to(self._device)
        attention_masks = tokenized_texts["attention_mask"].to(self._device)

        dataset = TensorDataset(input_ids, attention_masks)
        dataloader = DataLoader(dataset, batch_size=self.max_batch_size, shuffle=False)

        with torch.no_grad():
            for batch in dataloader:
                batch_input_ids, batch_attention_masks = batch
                encodings = self.minimal_batch_input_ids_to_early_embeddings(batch_input_ids)
                encodings = encodings * batch_attention_masks.unsqueeze(-1)
                for i in range(encodings.shape[0]):
                    embeddings.append(encodings[i:i + 1, :, :])

        embedding = torch.cat(embeddings, dim=0)
        assert embedding.shape == (len(texts), self.max_sequence_length,
                                   self.embedding_length), f"Batch embedding has wrong shape: {embedding.shape}"
        return embedding

    def late_embedding_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Performs early embedding but for a whole batch efficiently.
        As late embeddings are always pooled, you get 1 vector per text

        :param texts: Texts to encode
        :return: Tensor of size (batch_size, 1, embedding_dim)
        """
        assert len(texts) > 0, "No texts to encode"
        assert all(text and len(text) > 0 for text in texts), "Texts can not be empty"

        embeddings = []

        # Attempt to retrieve cached embeddings
        for text in texts:
            embedding = self._retrieve_late_embedding(text)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                embeddings.append(None)

        # Process texts that aren't cached
        texts_to_embed = [text for text, embedding in zip(texts, embeddings) if embedding is None]

        if not texts_to_embed:
            embedding = torch.cat(embeddings, dim=0)
            assert embedding.shape == (len(texts), 1,
                                       self.embedding_length), f"Batch embedding has wrong shape: {embedding.shape}"
            return embedding

        tokenized_texts = self._tokenizer.batch_encode_plus(
            texts_to_embed, return_tensors="pt", max_length=self.max_sequence_length, truncation=True,
            padding='max_length', add_special_tokens=False
        )

        input_ids = tokenized_texts["input_ids"].to(self._device)
        attention_masks = tokenized_texts["attention_mask"].to(self._device)

        dataset = TensorDataset(input_ids, attention_masks)
        dataloader = DataLoader(dataset, batch_size=self.max_batch_size, shuffle=False)

        pooled_embeddings_list = []
        with torch.no_grad():
            for batch in dataloader:
                batch_input_ids, batch_attention_masks = batch
                encodings = self.minimal_batch_input_ids_to_late_embeddings(batch_input_ids,
                                                                            batch_attention_masks=batch_attention_masks)
                mask = batch_attention_masks.unsqueeze(-1).expand_as(encodings)
                encodings = encodings * mask
                pooled_embeddings = self.pooling_strategy(encodings, dim=1).unsqueeze(1)
                for i in range(pooled_embeddings.shape[0]):
                    pooled_embeddings_list.append(pooled_embeddings[i:i + 1, :, :])

        # Update the embeddings list and cache new ones
        idx = 0
        for i, embedding in enumerate(embeddings):
            if embedding is None:
                embeddings[i] = pooled_embeddings_list[idx]
                self._cache_late_embedding(texts_to_embed[idx], embeddings[i])
                idx += 1

        embedding = torch.cat(embeddings, dim=0)
        assert embedding.shape == (
            len(texts), 1, self.embedding_length), f"Batch embedding has wrong shape: {embedding.shape}"
        return embedding

    def predict(self, inputs_embeds: torch.Tensor, past_key_values=None, attention_mask=None):
        """
        Predicts a single next token based on input embeddings
        :param inputs_embeds: Tensor(sequence_length, embedding_dim)
        :param past_key_values: If previous model state is passed, input_embeds should just be one token
        :return:
        """
        self._model.eval()
        embeddings = inputs_embeds
        mask = attention_mask
        if not past_key_values:
            # TODO: If dimensionality changes, also change attention_mask
            embeddings = self.add_special_token_embeddings(inputs_embeds)
            embeddings = self.add_positional_encoding(embeddings)

            if attention_mask is not None:
                batch_size, X, _ = embeddings.size()
                _, Y = attention_mask.size()

                if Y == X - 1:
                    ones = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
                    if self.tokenizer.padding_side == "left":
                        mask = torch.cat((attention_mask, ones), dim=1)
                    elif self.tokenizer.padding_side == "right":
                        mask = torch.cat((ones, attention_mask), dim=1)

        return self._model(inputs_embeds=embeddings, past_key_values=past_key_values, attention_mask=mask)

    def predict_token(self, inputs_embeds: torch.Tensor, past_key_values=None, sample_from_top=1) -> Dict:
        """
        Predict a single token
        :param inputs_embeds: Tensor(sequence_length, embedding_dim)
        :param past_key_values: If previous model state is passed, input_embeds should just be one token
        :param sample_from_top: Number of top logits to sample from
        :return:
        """
        output = self.predict(inputs_embeds, past_key_values=past_key_values)
        last_logits = output.logits[:, -1, :]

        # Get the top 'sample_from_top' logits and their indices
        top_logits, top_indices = torch.topk(last_logits, sample_from_top, dim=-1)

        # Randomly select from the top logits for each item in the batch
        selected_indices = torch.tensor([indices[random.randint(0, sample_from_top - 1)] for indices in top_indices]).to(self._device)

        # Convert the selected token ID to a string
        # Note: This will only convert the first token in the batch for simplicity
        token_str = self._tokenizer.convert_ids_to_tokens([selected_indices[0].item()])[0]
        if token_str.startswith(self.space_token):
            token_str = f" {token_str[1:]}"

        return {
            "output": output,
            "token_id": selected_indices,
            "token_str": token_str,
        }

    def predict_sequence(self, input_embeds: torch.Tensor, stop_tokens=None, n_tokens=64, sample_from_top=1) -> Dict:
        """
        Predicts a sequence of tokens until eos or stop token is received, or max sequence length is reached
        :param input_embeds: Tensor(sequence_length, embedding_dim)
        :param stop_tokens: List of stop tokens
        :param n_tokens: Maximum numbers of tokens to generate
        :return:
        """
        past_key_values = None
        output_string = ""
        outputs = []
        if stop_tokens is None:
            stop_tokens = []
        for i in range(n_tokens):
            prediction = self.predict_token(input_embeds, past_key_values=past_key_values, sample_from_top=sample_from_top)
            output = prediction["output"]
            outputs.append(output)
            past_key_values = output.past_key_values
            token_str = prediction["token_str"].replace(self.space_token, " ")
            token_id = prediction["token_id"]
            if token_str in stop_tokens or token_id == self._tokenizer.eos_token_id:
                break
            output_string += token_str
            token_embedding = self.token_ids_to_embeddings(token_id.unsqueeze(0))
            input_embeds = token_embedding
        return {
            "output_string": output_string,
            "outputs": outputs,
        }

    def predict_top_k(self, inputs_embeds: torch.Tensor, target_id: int, k=5, past_key_values=None):
        """
        Predicts whether a given token (target_id) was likely to be predicted by model with top-k likelihood
        :param inputs_embeds: Tensor(sequence_length, embedding_dim)
        :param target_id: Target token ID
        :param k: top-k likelihood
        :param past_key_values: If previous model state is passed, input_embeds should just be one token
        :return:
        """
        output = self.predict(inputs_embeds, past_key_values=past_key_values)
        logits = output.logits[:, -1, :]  # Getting the logits of the last token in the sequence
        probabilities = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probabilities, k, sorted=True)
        top_k_tokens = [idx.item() for idx in top_k_indices[0]]
        target_in_top_k = target_id in top_k_tokens
        target_k = top_k_tokens.index(target_id) + 1 if target_in_top_k else None
        result = {
            'output': output,
            'target_in_top_k': target_in_top_k,
            'target_k': target_k,
            'top_k_token_ids': top_k_tokens,
            'top_k_tokens': self._tokenizer.convert_ids_to_tokens(top_k_tokens),
            'top_k_probs': top_k_probs[0].tolist(),
            'target_prob': probabilities[0, target_id].item(),
        }

        return result

    def predict_top_k_sequence(self, input_embeds: torch.Tensor, target_token_ids: Optional[List[int]] = None,
                               source_text: Optional[str] = None, target_text: Optional[str] = None, k=5) -> Dict:
        """
        Predicts a sequence with top-k probability for each token
        :param input_embeds: Tensor(sequence_length, embedding_dim)
        :param target_token_ids: Token IDs that shall be predicted. Can be omitted when source and target text are provided instead
        :param source_text: Text that is taken as base for prediction
        :param target_text: Text that shall be predicted
        :param k: top-k likelihood
        :return:
        """
        probability = 1
        is_top_k = True
        past_key_values = None
        outputs = []
        target_k = 1

        if target_token_ids is None:
            assert source_text and target_text, "When target_token_ids is not provided, you must provide source_text and target_text instead"
            source_text = source_text.strip()
            target_text = f" {target_text.strip()}"
            n_source_tokens = self.text_to_input_ids(source_text).shape[1]
            target_token_ids = self.text_to_input_ids(source_text + target_text).cpu().numpy().flatten().tolist()[
                               n_source_tokens:]
        else:
            if source_text or target_text:
                logging.warning(
                    "You provided both target texts and tokens. Text will be ignored and tokens be used instead")

        for target_token_id in target_token_ids:
            top_k_prediction = self.predict_top_k(input_embeds, target_token_id, k=k, past_key_values=past_key_values)
            target_in_top_k = top_k_prediction['target_in_top_k']
            target_prob = top_k_prediction['target_prob']
            output = top_k_prediction['output']
            past_key_values = output.past_key_values
            outputs.append(output)
            probability *= target_prob
            if not target_in_top_k:
                target_k = None
                is_top_k = False
            if target_in_top_k and target_k is not None:
                target_k = max(target_k, top_k_prediction['target_k'])
            token_embedding = self.token_ids_to_embeddings(
                torch.Tensor([[target_token_id]]).to(torch.int).to(self._device))
            input_embeds = token_embedding
        return {
            "outputs": outputs,
            "is_top_k": is_top_k,
            "target_k": target_k,
            "probability": probability,
            "target_token_ids": target_token_ids,
        }

    def decode_input_embedding(self, input_embeds: torch.Tensor) -> str:
        """
        Converts an imput embedding sequence back to a string sequence
        :param input_embeds: Torch.tensor(sequence_length, embedding_dim)
        :return: str
        """
        assert input_embeds.dim() == 2, "The input must be a single sequence, not a batch"
        assert input_embeds.shape[1] == self.embedding_length, "Embeddings dimension incorrect for this llm"
        # Normalize the input embeddings
        input_embeds = input_embeds / input_embeds.norm(dim=-1, keepdim=True)

        # Get and normalize the embeddings from the model
        model_embeddings = self._model.get_input_embeddings().weight
        model_embeddings = model_embeddings / model_embeddings.norm(dim=-1, keepdim=True)

        # Calculate cosine similarities
        similarities = torch.matmul(model_embeddings, input_embeds.T)

        # Find the indices of the maximum values in similarities
        max_indices = torch.argmax(similarities, dim=0)

        # Decode the tokens
        tokens = self._tokenizer.convert_ids_to_tokens(max_indices.tolist())

        # Join tokens into a string
        decoded_string = self._tokenizer.convert_tokens_to_string(tokens)

        return decoded_string

    @property
    def model(self) -> PreTrainedModel:
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    @property
    def embedding_length(self) -> int:
        return self._embedding_length
