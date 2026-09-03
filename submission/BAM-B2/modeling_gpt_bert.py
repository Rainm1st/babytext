import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput, MaskedLMOutput
from transformers.utils import ModelOutput

try:
    from .configuration_gpt_bert import GPTBertConfig
except ImportError:
    from configuration_gpt_bert import GPTBertConfig


@dataclass
class GPTBertTrainingOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    ce_loss: Optional[torch.Tensor] = None
    z_loss: Optional[torch.Tensor] = None
    accuracy: Optional[torch.Tensor] = None
    num_tokens: Optional[torch.Tensor] = None


class GeGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = x.chunk(2, dim=-1)
        return value * F.gelu(gate, approximate="tanh")


def _relative_position_buckets(
    relative_position: torch.Tensor,
    bucket_size: int,
    max_position: int,
) -> torch.Tensor:
    sign = torch.sign(relative_position)
    mid = bucket_size // 2
    abs_pos = torch.where(
        (relative_position < mid) & (relative_position > -mid),
        torch.full_like(relative_position, mid - 1),
        torch.abs(relative_position).clamp(max=max_position - 1),
    )
    safe = abs_pos.clamp(min=mid)
    log_pos = (
        torch.ceil(
            torch.log(safe.float() / mid)
            / math.log((max_position - 1) / mid)
            * (mid - 1)
        ).long()
        + mid
    )
    bucket_pos = torch.where(abs_pos <= mid, relative_position, log_pos * sign)
    return bucket_size - 1 + bucket_pos.long()


class GPTBertEmbeddings(nn.Module):
    def __init__(self, config: GPTBertConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.word_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            elementwise_affine=False,
        )
        self.relative_embeddings = nn.Parameter(
            torch.empty(2 * config.position_bucket_size - 1, config.hidden_size)
        )
        self.relative_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(
            self.word_embeddings.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
        )
        nn.init.trunc_normal_(
            self.relative_embeddings, mean=0.0, std=std, a=-2 * std, b=2 * std
        )
        if self.word_embeddings.padding_idx is not None:
            with torch.no_grad():
                self.word_embeddings.weight[self.word_embeddings.padding_idx].zero_()

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.dropout(self.word_norm(self.word_embeddings(input_ids)))
        rel = self.relative_norm(self.relative_embeddings)
        return x, rel


class GPTBertAttention(nn.Module):
    def __init__(self, config: GPTBertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.qk_proj = nn.Linear(config.hidden_size, 2 * config.hidden_size)
        self.vg_proj = nn.Linear(config.hidden_size, 2 * config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.pre_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            elementwise_affine=False,
        )
        self.post_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            elementwise_affine=False,
        )
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.scale = 1.0 / math.sqrt(3.0 * self.head_dim)

        positions = (
            torch.arange(config.max_position_embeddings).unsqueeze(1)
            - torch.arange(config.max_position_embeddings).unsqueeze(0)
        )
        buckets = _relative_position_buckets(
            positions,
            config.position_bucket_size,
            config.max_position_embeddings,
        )
        self.register_buffer("position_indices", buckets, persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        for layer in (self.qk_proj, self.vg_proj, self.out_proj):
            nn.init.trunc_normal_(
                layer.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
            )
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        blocked_mask: torch.Tensor,
        relative_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        x = self.pre_norm(hidden_states)

        query, key = self.qk_proj(x).chunk(2, dim=-1)
        value, gate = self.vg_proj(x).chunk(2, dim=-1)
        gate = F.gelu(gate)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale

        rel_qk = self.qk_proj(self.dropout(relative_embeddings))
        rel_q, rel_k = rel_qk.chunk(2, dim=-1)
        indices = self.position_indices[:seq_len, :seq_len]
        rel_q = F.embedding(indices, rel_q).view(
            seq_len, seq_len, self.num_heads, self.head_dim
        )
        rel_k = F.embedding(indices, rel_k).view(
            seq_len, seq_len, self.num_heads, self.head_dim
        )

        scores = scores + torch.einsum(
            "bhqd,qkhd->bhqk", query, rel_k * self.scale
        )
        scores = scores + torch.einsum(
            "bhkd,qkhd->bhqk", key * self.scale, rel_q
        )

        scores = scores.masked_fill(blocked_mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        probs = self.dropout(probs)

        context = torch.matmul(probs, value)
        context = context.permute(0, 2, 1, 3).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        context = context * gate
        context = self.post_norm(context)
        context = self.out_proj(context)
        return self.out_dropout(context)


class GPTBertFeedForward(nn.Module):
    def __init__(self, config: GPTBertConfig, layer_index: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            elementwise_affine=False,
        )
        self.fc1 = nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,
            bias=False,
        )
        self.act = GeGLU()
        self.norm2 = nn.LayerNorm(
            config.intermediate_size,
            eps=config.layer_norm_eps,
            elementwise_affine=False,
        )
        self.fc2 = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.reset_parameters(layer_index)

    def reset_parameters(self, layer_index: int):
        std = math.sqrt(2.0 / (5.0 * self.fc2.out_features))
        nn.init.trunc_normal_(self.fc1.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.trunc_normal_(self.fc2.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        scale = math.sqrt(1.0 / (2.0 * (1 + layer_index)))
        with torch.no_grad():
            self.fc1.weight.mul_(scale)
            self.fc2.weight.mul_(scale)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.norm1(hidden_states)
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm2(x)
        x = self.fc2(x)
        return self.dropout(x)


class GPTBertLayer(nn.Module):
    def __init__(self, config: GPTBertConfig, layer_index: int):
        super().__init__()
        self.attention = GPTBertAttention(config)
        self.ffn = GPTBertFeedForward(config, layer_index)

    def forward(
        self,
        hidden_states: torch.Tensor,
        blocked_mask: torch.Tensor,
        relative_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(
            hidden_states, blocked_mask, relative_embeddings
        )
        hidden_states = hidden_states + self.ffn(hidden_states)
        return hidden_states


class GPTBertPreTrainedModel(PreTrainedModel):
    config_class = GPTBertConfig
    base_model_prefix = "gpt_bert"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        # Components initialize themselves to match the LTG/GPT-BERT recipe.
        return


class GPTBertModel(GPTBertPreTrainedModel):
    def __init__(self, config: GPTBertConfig):
        super().__init__(config)
        self.embeddings = GPTBertEmbeddings(config)
        self.layers = nn.ModuleList(
            [GPTBertLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _build_blocked_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, seq_len, device=input_ids.device, dtype=torch.long
            )
        key_padding = attention_mask.eq(0)[:, None, None, :]
        if is_causal:
            causal = torch.ones(
                seq_len, seq_len, device=input_ids.device, dtype=torch.bool
            ).triu(diagonal=1)[None, None, :, :]
            return key_padding | causal
        return key_padding.expand(batch_size, 1, seq_len, seq_len)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        blocked_mask = self._build_blocked_mask(
            input_ids, attention_mask, is_causal=is_causal
        )
        hidden_states, relative_embeddings = self.embeddings(input_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, blocked_mask, relative_embeddings
            )
        if not return_dict:
            return (hidden_states,)
        return BaseModelOutput(last_hidden_state=hidden_states)


class GPTBertLMHead(nn.Module):
    def __init__(self, config: GPTBertConfig, embedding_weight: nn.Parameter):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            elementwise_affine=False,
        )
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.decoder.weight = embedding_weight
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.reset_parameters(config.hidden_size)

    def reset_parameters(self, hidden_size: int):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(
            self.dense.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
        )
        nn.init.zeros_(self.dense.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.norm(x)
        return self.decoder(x) + self.bias


class GPTBertForMaskedLM(GPTBertPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight"]

    def __init__(self, config: GPTBertConfig):
        super().__init__(config)
        self.gpt_bert = GPTBertModel(config)
        self.lm_head = GPTBertLMHead(
            config, self.gpt_bert.embeddings.word_embeddings.weight
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.gpt_bert.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.gpt_bert.set_input_embeddings(value)
        self.lm_head.decoder.weight = value.weight

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def _selected_stats(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ):
        ce_loss = F.cross_entropy(logits.float(), labels)
        z_loss = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()
        return ce_loss, z_loss, accuracy

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        mode: str = "mntp",
        z_loss_weight: float = 0.0,
        return_dict: bool = True,
        **kwargs,
    ):
        if mode not in {"mntp", "masked", "causal"}:
            raise ValueError(f"Unsupported mode: {mode}")
        is_causal = mode == "causal"
        hidden = self.gpt_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            is_causal=is_causal,
            return_dict=True,
        ).last_hidden_state

        if labels is not None and mode in {"mntp", "masked"}:
            # MNTP: the hidden state at position i-1 predicts a masked token at i.
            valid = labels[:, 1:].ne(-100)
            selected_hidden = hidden[:, :-1][valid]
            selected_labels = labels[:, 1:][valid]
            if selected_labels.numel() == 0:
                raise RuntimeError("MNTP batch contains no prediction targets")
            selected_logits = self.lm_head(selected_hidden)
            ce_loss, z_loss, accuracy = self._selected_stats(
                selected_logits, selected_labels
            )
            loss = ce_loss + z_loss_weight * z_loss
            return GPTBertTrainingOutput(
                loss=loss,
                logits=None,
                ce_loss=ce_loss.detach(),
                z_loss=z_loss.detach(),
                accuracy=accuracy.detach(),
                num_tokens=torch.tensor(
                    selected_labels.numel(), device=input_ids.device
                ),
            )

        if labels is not None and mode == "causal":
            # Training input is [BOS] + tokens[:-1]; labels are tokens.
            logits = self.lm_head(hidden)
            valid = labels.ne(-100)
            selected_logits = logits[valid]
            selected_labels = labels[valid]
            ce_loss, z_loss, accuracy = self._selected_stats(
                selected_logits, selected_labels
            )
            loss = ce_loss + z_loss_weight * z_loss
            return GPTBertTrainingOutput(
                loss=loss,
                logits=None,
                ce_loss=ce_loss.detach(),
                z_loss=z_loss.detach(),
                accuracy=accuracy.detach(),
                num_tokens=torch.tensor(
                    selected_labels.numel(), device=input_ids.device
                ),
            )

        raw_logits = self.lm_head(hidden)
        # The BabyLM MNTP backend selects target_position - 1 itself.
        if not return_dict:
            return (raw_logits,)
        return MaskedLMOutput(logits=raw_logits)


class GPTBertForCausalLM(GPTBertForMaskedLM):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        hidden = self.gpt_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            is_causal=True,
            return_dict=True,
        ).last_hidden_state
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        return CausalLMOutput(loss=loss, logits=logits)
