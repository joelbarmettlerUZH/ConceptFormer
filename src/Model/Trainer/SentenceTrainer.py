import logging
from typing import List, Dict

import torch
import wandb
from datasets import Dataset
from torch import nn
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from src.Model.GraphAttentionEmbedder.GraphAttentionEmbedder import GraphAttentionEmbedder


class SentenceTrainer(nn.Module):
    def __init__(self, llm, graph_embedder: GraphAttentionEmbedder, replace_subject=False):
        super(SentenceTrainer, self).__init__()
        self.llm = llm
        self.graph_embedder = graph_embedder
        self.replace_subject = replace_subject

    def embed_sentence(
            self,
            sentence,
            subject_boundary_start,
            subject_boundary_end,
            object_boundary_start,
            object_boundary_end,
            graph_embedding,
            index=0
    ):
        source_text = sentence[:object_boundary_start]
        source_text = source_text.strip()

        target_text = sentence[object_boundary_start:object_boundary_end]
        target_text = f" {target_text.strip()}"

        logging.debug(f"{index}: SentenceTrainer.source_text: {source_text}")
        logging.debug(f"{index}: SentenceTrainer.target_text: {target_text}")

        n_source_tokens = self.llm.text_to_input_ids(source_text).shape[1] if source_text else 0
        target_token_ids = self.llm.text_to_input_ids(source_text + target_text).cpu().numpy().flatten().tolist()[
                           n_source_tokens:]

        logging.debug(f"{index}: SentenceTrainer.n_source_token_ids: {n_source_tokens}")
        logging.debug(f"{index}: SentenceTrainer.target_token_ids: {target_token_ids}")

        target_token_strings = self.llm.tokenizer.convert_ids_to_tokens(target_token_ids)

        logging.debug(f"{index}: SentenceTrainer.target_token_strings: {target_token_strings}")

        embedding = []

        # Start text is the part of text before the subject
        start_text = ""
        if subject_boundary_start > 0:
            if self.replace_subject:
                start_text = sentence[:subject_boundary_start]
            else:
                start_text = sentence[:subject_boundary_end]
            start_text = start_text.rstrip()
            logging.debug(f"{index}: SentenceTrainer.start_text: {start_text}")

        logging.debug(f"{index}: SentenceTrainer.subject: {sentence[subject_boundary_start:subject_boundary_end]}")

        end_text = sentence[subject_boundary_end:object_boundary_start].rstrip()

        if (not start_text) and (not end_text):
            start_token_ids = []
            end_token_ids = []
        elif not start_text:
            start_token_ids = []
            end_token_ids = self.llm.text_to_input_ids(end_text)
        elif not end_text:
            start_token_ids = self.llm.text_to_input_ids(start_text)
            end_token_ids = []
        else:
            full_text = f"{start_text.rstrip()} {end_text.lstrip()}"
            start_token_ids = self.llm.text_to_input_ids(start_text)
            n_start_text_tokens = start_token_ids.shape[1]
            end_token_ids = self.llm.text_to_input_ids(full_text)[:, n_start_text_tokens:]

        if start_text:
            start_embedding = self.llm.minimal_batch_input_ids_to_early_embeddings(start_token_ids)
            logging.debug(f"{index}: SentenceTrainer.start_embedding: {start_embedding.shape}")
            embedding.append(start_embedding)

        embedding.append(graph_embedding.unsqueeze(0))

        if end_text:
            end_embedding = self.llm.minimal_batch_input_ids_to_early_embeddings(end_token_ids)
            logging.debug(f"{index}: SentenceTrainer.end_text: {end_text}")
            logging.debug(f"{index}: SentenceTrainer.end_embedding: {end_embedding.shape}")
            embedding.append(end_embedding)

        return torch.cat(embedding, dim=1), target_token_ids

    def forward(
            self,
            sentences: List[str],
            subject_boundaries_start: List[int],
            subject_boundaries_end: List[int],
            object_boundaries_start: List[int],
            object_boundaries_end: List[int],
            central_node_embedding: torch.Tensor,
            node_embeddings: torch.Tensor,
            edge_embeddings: torch.Tensor
    ):
        device = central_node_embedding.device

        max_length = 0
        padding_side = self.llm.tokenizer.padding_side

        logging.debug(f"SentenceTrainer.central_node_embedding: {central_node_embedding.shape}")
        logging.debug(f"SentenceTrainer.node_embeddings: {node_embeddings.shape}")
        logging.debug(f"SentenceTrainer.edge_embeddings: {edge_embeddings.shape}")

        batch_input_embeds = []
        batch_attention_masks = []

        graph_embeddings = self.graph_embedder(central_node_embedding, node_embeddings, edge_embeddings)

        cache = []
        for index, (sentence,
                    subject_boundary_start,
                    subject_boundary_end,
                    object_boundary_start,
                    object_boundary_end,
                    ) in enumerate(
            zip(sentences,
                subject_boundaries_start,
                subject_boundaries_end,
                object_boundaries_start,
                object_boundaries_end)):
            input_embeds, target_token_ids = self.embed_sentence(
                sentence,
                subject_boundary_start,
                subject_boundary_end,
                object_boundary_start,
                object_boundary_end,
                graph_embeddings[index],
                index=index
            )
            cache.append([input_embeds, target_token_ids])
            max_length = max(max_length, input_embeds.size(1) + len(target_token_ids))

        for index in range(len(sentences)):
            input_embeds, target_token_ids = cache[index]
            logging.debug(f"{index}: SentenceTrainer.input_embeds: {input_embeds.shape}")

            seq_length = input_embeds.size(1)
            padding_length = max_length - seq_length
            padding = torch.zeros((input_embeds.size(0), padding_length, input_embeds.size(2)), device=device)
            attention_mask = torch.ones(max_length, device=device)

            if padding_side == 'right':
                padded_input_embeds = torch.cat((input_embeds, padding), dim=1)
                attention_mask[seq_length:] = 0
            else:
                padded_input_embeds = torch.cat((padding, input_embeds), dim=1)
                attention_mask[:padding_length] = 0

            batch_input_embeds.append(padded_input_embeds)
            batch_attention_masks.append(attention_mask)

            for target_token_id in target_token_ids[:-1]:
                token_ids = torch.Tensor([[target_token_id]]).to(torch.int).to(device)
                token_embedding = self.llm.token_ids_to_embeddings(token_ids)
                input_embeds = torch.cat((input_embeds, token_embedding), dim=1)

                seq_length = input_embeds.size(1)
                padding_length = max_length - seq_length
                padding = torch.zeros((input_embeds.size(0), padding_length, input_embeds.size(2)), device=device)
                attention_mask = torch.ones(max_length, device=device)

                if padding_side == 'right':
                    padded_input_embeds = torch.cat((input_embeds, padding), dim=1)
                    attention_mask[seq_length:] = 0
                else:
                    padded_input_embeds = torch.cat((padding, input_embeds), dim=1)
                    attention_mask[:padding_length] = 0

                batch_input_embeds.append(padded_input_embeds)
                batch_attention_masks.append(attention_mask)

        all_logits = []
        for i in range(0, len(batch_input_embeds), self.llm.max_batch_size):
            sub_input_embeds = torch.cat(batch_input_embeds[i:i + self.llm.max_batch_size], dim=0)
            sub_attention_masks = self.llm.attention_mask(batch_attention_masks[i:i + self.llm.max_batch_size])

            # Batch prediction
            output = self.llm.predict(sub_input_embeds, attention_mask=sub_attention_masks)
            logits = output.logits

            sub_logits = logits[:, -1, :]
            all_logits.append(sub_logits)

        all_logits = torch.cat(all_logits, dim=0)
        return all_logits

    def evaluate(
            self,
            test_dataset: Dataset,
            k: int,
            run: Run,
            embedding_lookup_table: Dict[str, torch.Tensor] | None = None,
            prefix='',
    ):
        device = self.llm._device

        hits = [0 for _ in range(k)]
        misses = [0 for _ in range(k)]

        examples_table = wandb.Table(columns=[
            "Source",
            "Target",
            "Prediction",
            "k",
            "Target k",
            "Is Hit",
            "Subject",
            "Predicate",
            "Object",
            "Subject ID",
            "Predicate ID",
            "Object ID",
            "Subject Rank",
            "Object Rank",
        ])

        for i in tqdm(range(len(test_dataset)), desc=f'Evaluating', unit='sentences'):
            data = test_dataset[i]
            sentence = data['sentence']
            strived_for_k = data.get('k', 1)
            _subject = data['subject_label']
            _predicate = data['predicate_label']
            _object = data['object_label']

            subject_boundary_start = data['subject_boundary_start']
            subject_boundary_end = data['subject_boundary_end']

            object_boundary_start = data['object_boundary_start']
            object_boundary_end = data['object_boundary_end']

            source_text = sentence[:object_boundary_start].strip()
            target_text = sentence[object_boundary_start:].strip()

            if embedding_lookup_table:
                graph_embeddings = embedding_lookup_table.get(data['subject_id'])
                if graph_embeddings is None:
                    continue
            else:
                central_node_embedding = data['central_node_embedding'].to(device, non_blocking=True).unsqueeze(0)
                node_embeddings = data['node_embeddings'].to(device, non_blocking=True).unsqueeze(0)
                edge_embeddings = data['edge_embeddings'].to(device, non_blocking=True).unsqueeze(0)
                graph_embeddings = self.graph_embedder(central_node_embedding, node_embeddings, edge_embeddings)
            input_embeds, target_token_ids = self.embed_sentence(
                sentence,
                subject_boundary_start,
                subject_boundary_end,
                object_boundary_start,
                object_boundary_end,
                graph_embeddings[0]
            )

            top_k_output = self.llm.predict_top_k_sequence(input_embeds, target_token_ids=target_token_ids, k=k)

            is_top_k = top_k_output['is_top_k']
            target_k = top_k_output['target_k']

            prediction = self.llm.predict_sequence(input_embeds, n_tokens=len(top_k_output['target_token_ids']))
            examples_table.add_data(
                source_text,
                target_text,
                prediction['output_string'],
                target_k,
                strived_for_k,
                bool(target_k and target_k <= strived_for_k),
                _subject,
                _predicate,
                _object,
                data['subject_id'],
                data['predicate_id'],
                data['object_id'],
                data['subject_rank'],
                data['object_rank'],
            )

            if not is_top_k:
                for i in range(len(misses)):
                    misses[i] += 1
            else:
                for i in range(target_k - 1):
                    misses[i] += 1

                for i in range(target_k - 1, len(misses)):
                    hits[i] += 1

        x_values = list(range(1, k + 1))
        y_values = list([hits[i] / (hits[i] + misses[i]) for i in range(k)])

        data = [[x, y] for (x, y) in zip(x_values, y_values)]
        topk_table = wandb.Table(data=data, columns=["x", "y"])
        wandb.log({
            f"{prefix}examples": examples_table,
            f"{prefix}plot": wandb.plot.line(topk_table, "x", "y", title=f"{prefix}TopK Hit Ratio"),
        })

        for k_i in [1, 2, 3, 4, 5, 10, 15, 25, 50]:
            wandb.log({
                f'{prefix}k{k_i}': hits[k_i - 1] / (hits[k_i - 1] + misses[k_i - 1])
            })
            wandb.summary[f'{prefix}k{k_i}'] = hits[k_i - 1] / (hits[k_i - 1] + misses[k_i - 1])
