"""ConceptFormer v2 model architecture.

Key changes from v1:
- Vectorised multi-head graph attention (no Python loop over heads)
- LayerNorm + residual connections
- Learnable entity/relation nn.Embeddings (replaces BigGraph)
- Full-sequence teacher forcing (replaces token-by-token forward)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from conceptformer.config import ConceptFormerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Learnable knowledge-graph embedding tables
# ---------------------------------------------------------------------------

class KGEmbeddings(nn.Module):
    """Learnable entity and relation embeddings that replace BigGraph.

    Embeddings are trained end-to-end with the rest of ConceptFormer.
    Initialised from a truncated normal so they live in a similar scale
    to the LLM's token embeddings.
    """

    def __init__(self, num_entities: int, num_relations: int, embed_dim: int):
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embed_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embed_dim)
        nn.init.trunc_normal_(self.entity_embeddings.weight, std=0.02)
        nn.init.trunc_normal_(self.relation_embeddings.weight, std=0.02)

    def get_node_embeddings(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """entity_ids: (batch, num_neighbors) or (batch,) -> (..., embed_dim)"""
        return self.entity_embeddings(entity_ids)

    def get_edge_embeddings(self, relation_ids: torch.Tensor) -> torch.Tensor:
        """relation_ids: (batch, num_neighbors) -> (..., embed_dim)"""
        return self.relation_embeddings(relation_ids)


# ---------------------------------------------------------------------------
# Modernised graph attention embedder
# ---------------------------------------------------------------------------

class GraphAttentionEmbedder(nn.Module):
    """Multi-head graph attention that produces ``d`` pseudo-word embeddings.

    Improvements over v1
    --------------------
    * All ``d`` heads are computed as a single batched matmul (no Python loop).
    * Pre-norm architecture: LayerNorm → Attention → Residual → LayerNorm → MLP → Residual.
    * Dropout on attention weights and MLP hidden activations.
    """

    def __init__(
        self,
        embed_dim: int,
        num_pseudo_words: int,
        hidden_dim: int,
        num_mlp_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d = num_pseudo_words
        self.embed_dim = embed_dim
        self.scale = embed_dim ** 0.5

        # --- Per-head projections (stacked into a single weight for efficiency) ---
        # Shape convention: we keep d as an explicit "head" dimension and
        # compute all heads simultaneously via einsum / bmm.
        self.W_q = nn.Parameter(torch.empty(num_pseudo_words, embed_dim, embed_dim))
        self.W_k = nn.Parameter(torch.empty(num_pseudo_words, embed_dim, embed_dim))
        self.W_v = nn.Parameter(torch.empty(num_pseudo_words, embed_dim, embed_dim))
        self.W_o = nn.Parameter(torch.empty(num_pseudo_words, embed_dim, embed_dim))
        for p in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(p.view(-1, embed_dim))  # per-head init

        # --- Norms ---
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.norm_mlp = nn.LayerNorm(embed_dim)

        # --- MLP ---
        layers = []
        in_dim = embed_dim
        for _ in range(num_mlp_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, embed_dim))
        layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

        self.attn_dropout = nn.Dropout(dropout)

        # Learnable query bias per pseudo-word (allows each head to
        # specialise even when attending to the same central node).
        self.query_bias = nn.Parameter(torch.zeros(num_pseudo_words, 1, embed_dim))

    def forward(
        self,
        central_node: torch.Tensor,    # (B, 1, E)
        neighbor_nodes: torch.Tensor,   # (B, N, E)
        edge_features: torch.Tensor,    # (B, N, E)
        neighbor_mask: Optional[torch.Tensor] = None,  # (B, N) bool, True = valid
    ) -> torch.Tensor:
        """Returns (B, d, E) – one pseudo-word embedding per head."""
        B, N, E = neighbor_nodes.shape

        # --- Pre-norm ---
        central_normed = self.norm_attn(central_node)   # (B, 1, E)
        neighbor_normed = self.norm_attn(neighbor_nodes) # (B, N, E)

        # Expand central node to all d heads: (B, 1, E) -> (d, B, 1, E)
        cen = central_normed.unsqueeze(0).expand(self.d, -1, -1, -1)
        # Queries: (d, B, 1, E) x (d, E, E) -> (d, B, 1, E)
        Q = torch.einsum("dbse,deo->dbso", cen, self.W_q) + self.query_bias.unsqueeze(1)

        # Keys incorporate edge features (same as v1: K = W_k(neighbor) + edge)
        nei = neighbor_normed.unsqueeze(0).expand(self.d, -1, -1, -1)  # (d, B, N, E)
        edg = edge_features.unsqueeze(0).expand(self.d, -1, -1, -1)    # (d, B, N, E)
        K = torch.einsum("dbne,deo->dbno", nei, self.W_k) + edg        # (d, B, N, E)

        # Values
        V = torch.einsum("dbne,deo->dbno", nei, self.W_v)              # (d, B, N, E)

        # Scaled dot-product attention: (d, B, 1, E) x (d, B, E, N) -> (d, B, 1, N)
        attn_scores = torch.einsum("dbse,dbne->dbsn", Q, K) / self.scale

        # Mask out padding neighbors
        if neighbor_mask is not None:
            # neighbor_mask: (B, N) -> (1, B, 1, N)
            mask = neighbor_mask.unsqueeze(0).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)  # (d, B, 1, N)
        attn_probs = self.attn_dropout(attn_probs)

        # Weighted sum: (d, B, 1, N) x (d, B, N, E) -> (d, B, 1, E)
        context = torch.einsum("dbsn,dbne->dbse", attn_probs, V)

        # Output projection
        context = torch.einsum("dbse,deo->dbso", context, self.W_o)    # (d, B, 1, E)

        # Reshape: (d, B, 1, E) -> (B, d, E) and add residual from central node
        context = context.squeeze(2).permute(1, 0, 2)                  # (B, d, E)
        # Residual: broadcast central (B, 1, E) -> (B, d, E)
        context = context + central_node.expand(-1, self.d, -1)

        # --- MLP block with pre-norm + residual ---
        residual = context
        context = self.norm_mlp(context)
        context = self.mlp(context) + residual  # (B, d, E)

        return context


# ---------------------------------------------------------------------------
# Full ConceptFormer model
# ---------------------------------------------------------------------------

class ConceptFormerModel(nn.Module):
    """End-to-end ConceptFormer: graph attention → pseudo-words → LLM next-token prediction.

    The LLM backbone is frozen. Only the graph attention head and
    entity/relation embeddings are trained.

    Forward pass (teacher forcing)
    ------------------------------
    1. Look up entity/relation embeddings for the graph neighbourhood.
    2. Run ``GraphAttentionEmbedder`` to produce ``d`` pseudo-word vectors.
    3. Build the input embedding sequence:
       ``[start_text_embeds | pseudo_words | end_text_embeds | target_text_embeds]``
    4. Run a single forward pass through the frozen LLM.
    5. Compute cross-entropy loss only on the target positions.
    """

    def __init__(self, config: ConceptFormerConfig, llm, tokenizer):
        super().__init__()
        self.config = config
        self.llm = llm          # frozen HuggingFace CausalLM
        self.tokenizer = tokenizer

        embed_dim = llm.config.hidden_size
        config.embed_dim = embed_dim

        # Freeze LLM
        for param in llm.parameters():
            param.requires_grad = False

        # Trainable components
        self.kg_embeddings = KGEmbeddings(
            num_entities=config.entity_vocab_size,
            num_relations=config.relation_vocab_size,
            embed_dim=embed_dim,
        )

        hidden_dim = int(embed_dim * config.graph_head_width_multiplier)
        self.graph_attention = GraphAttentionEmbedder(
            embed_dim=embed_dim,
            num_pseudo_words=config.num_pseudo_words,
            hidden_dim=hidden_dim,
            num_mlp_layers=config.graph_head_layers,
            dropout=config.graph_head_dropout,
        )

        # Projection to align graph-attention output to LLM embedding space
        # (only if the dimensions diverge, otherwise identity)
        self.proj = nn.Identity()

    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings from the frozen LLM's embedding layer."""
        return self.llm.get_input_embeddings()(input_ids)

    def forward(
        self,
        # Graph inputs (integer IDs for embedding lookup)
        central_entity_ids: torch.Tensor,   # (B,)
        neighbor_entity_ids: torch.Tensor,   # (B, N)
        relation_ids: torch.Tensor,          # (B, N)
        neighbor_mask: torch.Tensor,         # (B, N) bool
        # Text inputs (token IDs)
        start_input_ids: torch.Tensor,       # (B, S_start) — text before subject
        start_attention_mask: torch.Tensor,  # (B, S_start)
        end_input_ids: torch.Tensor,         # (B, S_end) — text between subject and object
        end_attention_mask: torch.Tensor,    # (B, S_end)
        target_input_ids: torch.Tensor,      # (B, S_target) — the object text (labels)
        target_attention_mask: torch.Tensor, # (B, S_target)
    ) -> dict:
        """Full forward pass with teacher forcing. Returns dict with 'loss' and 'logits'."""
        B = central_entity_ids.size(0)
        device = central_entity_ids.device

        # 1. Graph embeddings
        central_emb = self.kg_embeddings.get_node_embeddings(central_entity_ids).unsqueeze(1)  # (B, 1, E)
        neighbor_emb = self.kg_embeddings.get_node_embeddings(neighbor_entity_ids)              # (B, N, E)
        edge_emb = self.kg_embeddings.get_edge_embeddings(relation_ids)                         # (B, N, E)

        # 2. Graph attention -> pseudo-words
        pseudo_words = self.graph_attention(central_emb, neighbor_emb, edge_emb, neighbor_mask)  # (B, d, E)
        pseudo_words = self.proj(pseudo_words)

        # 3. Get text embeddings from frozen LLM
        start_embeds = self.get_text_embeddings(start_input_ids)    # (B, S_start, E)
        end_embeds = self.get_text_embeddings(end_input_ids)        # (B, S_end, E)
        target_embeds = self.get_text_embeddings(target_input_ids)  # (B, S_target, E)

        # 4. Concatenate: [start | pseudo_words | end | target]
        inputs_embeds = torch.cat([start_embeds, pseudo_words, end_embeds, target_embeds], dim=1)

        # Build attention mask: all pseudo-words are attended to
        pseudo_mask = torch.ones(B, self.config.num_pseudo_words, dtype=torch.long, device=device)
        attention_mask = torch.cat([start_attention_mask, pseudo_mask, end_attention_mask, target_attention_mask], dim=1)

        # 5. Forward through frozen LLM
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs.logits  # (B, total_seq_len, vocab_size)

        # 6. Compute loss on target positions only (teacher forcing)
        # The target tokens are at the end of the sequence.
        # For causal LM: logits at position i predict token at position i+1.
        # So we want logits at positions [-(S_target+1) : -1] to predict target_input_ids.
        S_target = target_input_ids.size(1)
        # Shift: logits[..., :-1, :] predicts labels[..., 1:]
        # We only care about the last S_target tokens as labels.
        total_len = logits.size(1)
        target_start = total_len - S_target

        # logits that predict each target token: positions [target_start-1 .. total_len-2]
        shift_logits = logits[:, target_start - 1 : total_len - 1, :]  # (B, S_target, V)
        shift_labels = target_input_ids.clone()                         # (B, S_target)

        # Mask out padding in labels (set to -100 so CE ignores them)
        shift_labels[target_attention_mask == 0] = -100

        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )

        return {"loss": loss, "logits": shift_logits}

    def generate_concept_vectors(
        self,
        central_entity_ids: torch.Tensor,
        neighbor_entity_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Generate pseudo-word concept vectors without text context (for lookup tables)."""
        central_emb = self.kg_embeddings.get_node_embeddings(central_entity_ids).unsqueeze(1)
        neighbor_emb = self.kg_embeddings.get_node_embeddings(neighbor_entity_ids)
        edge_emb = self.kg_embeddings.get_edge_embeddings(relation_ids)
        pseudo_words = self.graph_attention(central_emb, neighbor_emb, edge_emb, neighbor_mask)
        return self.proj(pseudo_words)

    def predict_with_concept(
        self,
        pseudo_words: torch.Tensor,         # (1, d, E)
        start_input_ids: torch.Tensor,       # (1, S_start)
        end_input_ids: torch.Tensor,         # (1, S_end)
        max_new_tokens: int = 32,
        top_k: int = 50,
    ) -> dict:
        """Autoregressively generate tokens after injecting concept vectors.

        Used for evaluation / inference (not training).
        """
        start_embeds = self.get_text_embeddings(start_input_ids)
        end_embeds = self.get_text_embeddings(end_input_ids)

        inputs_embeds = torch.cat([start_embeds, pseudo_words, end_embeds], dim=1)
        seq_len = inputs_embeds.size(1)
        attention_mask = torch.ones(1, seq_len, device=inputs_embeds.device, dtype=torch.long)

        generated_ids = []
        past_key_values = None

        for step in range(max_new_tokens):
            if past_key_values is None:
                out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            else:
                out = self.llm(inputs_embeds=next_embeds, past_key_values=past_key_values, attention_mask=attention_mask)

            past_key_values = out.past_key_values
            next_logits = out.logits[:, -1, :]

            # Top-k sampling (greedy when top_k=1)
            if top_k == 1:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                top_vals, top_idx = torch.topk(next_logits, top_k, dim=-1)
                probs = F.softmax(top_vals, dim=-1)
                sampled = torch.multinomial(probs, 1)
                next_token = top_idx.gather(-1, sampled)

            generated_ids.append(next_token.squeeze(-1))

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            next_embeds = self.get_text_embeddings(next_token)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(1, 1, device=attention_mask.device, dtype=torch.long)], dim=1
            )

        generated_ids = torch.stack(generated_ids, dim=1) if generated_ids else torch.empty(1, 0, dtype=torch.long)
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return {"generated_ids": generated_ids, "text": text}

    def compute_top_k_accuracy(
        self,
        pseudo_words: torch.Tensor,         # (1, d, E)
        start_input_ids: torch.Tensor,       # (1, S_start)
        end_input_ids: torch.Tensor,         # (1, S_end)
        target_input_ids: torch.Tensor,      # (1, S_target)
        k: int = 50,
    ) -> dict:
        """Evaluate whether target tokens appear in top-k predictions.

        Returns the worst (highest) rank across all target tokens, or None if
        any target token is not in top-k.
        """
        start_embeds = self.get_text_embeddings(start_input_ids)
        end_embeds = self.get_text_embeddings(end_input_ids)
        target_embeds = self.get_text_embeddings(target_input_ids)

        inputs_embeds = torch.cat([start_embeds, pseudo_words, end_embeds], dim=1)

        worst_rank = 0
        all_in_top_k = True
        past_key_values = None

        target_ids = target_input_ids[0].tolist()

        for i, target_id in enumerate(target_ids):
            if past_key_values is None:
                out = self.llm(inputs_embeds=inputs_embeds)
            else:
                out = self.llm(inputs_embeds=next_embeds, past_key_values=past_key_values)

            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :]
            top_k_ids = torch.topk(logits, k, dim=-1).indices[0].tolist()

            if target_id in top_k_ids:
                rank = top_k_ids.index(target_id) + 1
                worst_rank = max(worst_rank, rank)
            else:
                all_in_top_k = False
                worst_rank = None
                break

            next_embeds = target_embeds[:, i : i + 1, :]

        return {
            "is_top_k": all_in_top_k,
            "target_k": worst_rank,
        }
