# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional

import numpy as np
import torch
import math
import transformer_engine as te
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformer_engine.pytorch.attention import DotProductAttention, apply_rotary_pos_emb

# ---------------------- Feed Forward Network -----------------------
class GPT2FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

        self._layer_id = None
        self._dim = d_model
        self._hidden_dim = d_ff
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._dim)
        torch.nn.init.trunc_normal_(self.layer1.weight, std=std, a=-3 * std, b=3 * std)

        # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
        std = 1.0 / math.sqrt(self._hidden_dim)
        if self._layer_id is not None:
            std = std / math.sqrt(2 * (self._layer_id + 1))
        torch.nn.init.trunc_normal_(self.layer2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)

        def activation_layer2_forward(x):
            x = self.activation(x)
            x = self.layer2(x)
            return x

        x = checkpoint(activation_layer2_forward, x, use_reentrant=False)
        return x


# ---------------------- Normalization Layer -----------------------
def get_normalization(name: str, channels: int):
    if name == "I":
        return nn.Identity()
    elif name == "R":
        return te.pytorch.RMSNorm(channels, eps=1e-6)
    else:
        raise ValueError(f"Normalization {name} not found")


class BaseAttentionOp(nn.Module):
    def __init__(self):
        super().__init__()


def torch_attention_op(q_B_S_H_D, k_B_S_H_D, v_B_S_H_D):
    """Computes multi-head attention using PyTorch's native implementation.

    This function provides a PyTorch backend alternative to Transformer Engine's attention operation.
    It rearranges the input tensors to match PyTorch's expected format, computes scaled dot-product
    attention, and rearranges the output back to the original format.

    The input tensor names use the following dimension conventions:

    - B: batch size
    - S: sequence length
    - H: number of attention heads
    - D: head dimension

    Args:
        q_B_S_H_D: Query tensor with shape (batch, seq_len, n_heads, head_dim)
        k_B_S_H_D: Key tensor with shape (batch, seq_len, n_heads, head_dim)
        v_B_S_H_D: Value tensor with shape (batch, seq_len, n_heads, head_dim)

    Returns:
        Attention output tensor with shape (batch, seq_len, n_heads * head_dim)
    """
    in_q_shape = q_B_S_H_D.shape
    in_k_shape = k_B_S_H_D.shape
    q_B_H_S_D = rearrange(q_B_S_H_D, "b ... h k -> b h ... k").view(in_q_shape[0], in_q_shape[-2], -1, in_q_shape[-1])
    k_B_H_S_D = rearrange(k_B_S_H_D, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
    v_B_H_S_D = rearrange(v_B_S_H_D, "b ... h v -> b h ... v").view(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
    result_B_S_HD = rearrange(
        torch.nn.functional.scaled_dot_product_attention(q_B_H_S_D, k_B_H_S_D, v_B_H_S_D), "b h ... l -> b ... (h l)"
    )

    return result_B_S_HD



class Attention(nn.Module):
    """
    A flexible attention module supporting both self-attention and cross-attention mechanisms.

    This module implements a multi-head attention layer that can operate in either self-attention
    or cross-attention mode. The mode is determined by whether a context dimension is provided.
    The implementation uses scaled dot-product attention and supports optional bias terms and
    dropout regularization.

    Args:
        query_dim (int): The dimensionality of the query vectors.
        context_dim (int, optional): The dimensionality of the context (key/value) vectors.
            If None, the module operates in self-attention mode using query_dim. Default: None
        n_heads (int, optional): Number of attention heads for multi-head attention. Default: 8
        head_dim (int, optional): The dimension of each attention head. Default: 64
        dropout (float, optional): Dropout probability applied to the output. Default: 0.0
        qkv_format (str, optional): Format specification for QKV tensors. Default: "bshd"
        backend (str, optional): Backend to use for the attention operation. Default: "transformer_engine"

    Examples:
        >>> # Self-attention with 512 dimensions and 8 heads
        >>> self_attn = Attention(query_dim=512)
        >>> x = torch.randn(32, 16, 512)  # (batch_size, seq_len, dim)
        >>> out = self_attn(x)  # (32, 16, 512)

        >>> # Cross-attention
        >>> cross_attn = Attention(query_dim=512, context_dim=256)
        >>> query = torch.randn(32, 16, 512)
        >>> context = torch.randn(32, 8, 256)
        >>> out = cross_attn(query, context)  # (32, 16, 512)
    """

    def __init__(
        self,
        query_dim: int,
        context_dim=None,
        n_heads=8,
        head_dim=64,
        dropout=0.0,
        qkv_format: str = "bshd",
        backend: str = "transformer_engine",
    ) -> None:
        super().__init__()
        self.is_selfattn = context_dim is None  # self attention

        assert backend in ["transformer_engine", "torch"], f"Invalid backend: {backend}"
        self.backend = backend

        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv_format = qkv_format
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = te.pytorch.RMSNorm(self.head_dim, eps=1e-6)

        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = te.pytorch.RMSNorm(self.head_dim, eps=1e-6)

        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_norm = nn.Identity()

        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()

        if self.backend == "transformer_engine":
            self.attn_op = DotProductAttention(
                self.n_heads,
                self.head_dim,
                num_gqa_groups=self.n_heads,
                attention_dropout=0,
                qkv_format=qkv_format,
                attn_mask_type="no_mask",
            )
        elif self.backend == "torch":
            self.attn_op = torch_attention_op

        self._query_dim = query_dim
        self._context_dim = context_dim
        self._inner_dim = inner_dim
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._query_dim)
        torch.nn.init.trunc_normal_(self.q_proj.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self._context_dim)
        torch.nn.init.trunc_normal_(self.k_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.v_proj.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self._inner_dim)
        torch.nn.init.trunc_normal_(self.output_proj.weight, std=std, a=-3 * std, b=3 * std)

        for layer in self.q_norm, self.k_norm, self.v_norm:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def compute_qkv(self, x, context=None, rope_emb=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(x)
        context = x if context is None else context
        k = self.k_proj(context)
        v = self.v_proj(context)
        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (q, k, v),
        )

        def apply_norm_and_rotary_pos_emb(q, k, v, rope_emb):
            q = self.q_norm(q)
            k = self.k_norm(k)
            v = self.v_norm(v)
            if self.is_selfattn and rope_emb is not None:  # only apply to self-attention!
                q = apply_rotary_pos_emb(q, rope_emb, tensor_format=self.qkv_format, fused=True)
                k = apply_rotary_pos_emb(k, rope_emb, tensor_format=self.qkv_format, fused=True)
            return q, k, v

        q, k, v = checkpoint(apply_norm_and_rotary_pos_emb, q, k, v, rope_emb, use_reentrant=False)

        return q, k, v

    def compute_attention(self, q, k, v):
        result = self.attn_op(q, k, v)  # [B, S, H, D]
        return self.output_dropout(self.output_proj(result))

    def forward(
        self,
        x,
        context=None,
        rope_emb=None,
    ):
        """
        Args:
            x (Tensor): The query tensor of shape [B, Mq, K]
            context (Optional[Tensor]): The key tensor of shape [B, Mk, K] or use x as context [self attention] if None
        """
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        return self.compute_attention(q, k, v)

    def set_context_parallel_group(self, process_group, ranks, stream):
        self.attn_op.set_context_parallel_group(process_group, ranks, stream)
