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

import functools
import math
from typing import Optional

import numpy as np
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
import numpy as np
import torch
from einops.layers.torch import Rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint

from cosmos_predict2.diffusion.module.attention import Attention, GPT2FeedForward
from cosmos_predict2.utils import log

try:
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

    def create_policy_fn():
        def policy_fn(ctx, op, *args, **kwargs):
            op_name = str(op)
            if (
                op_name == "aten.add.default"
                or op_name == "aten.mul.Tensor"
                or op_name == "aten.native_layer_norm.default"
            ):
                return CheckpointPolicy.PREFER_RECOMPUTE

            return CheckpointPolicy.MUST_SAVE

        return policy_fn

    _context_fn = functools.partial(create_selective_checkpoint_contexts, create_policy_fn())
except ImportError:
    _context_fn = None


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Timesteps(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T):
        assert timesteps_B_T.ndim == 2, f"Expected 2D input, got {timesteps_B_T.ndim}"
        in_dype = timesteps_B_T.dtype
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return rearrange(emb.to(dtype=in_dype), "(b t) d -> b t d", b=timesteps_B_T.shape[0], t=timesteps_B_T.shape[1])


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_adaln_lora: bool = False):
        super().__init__()
        log.debug(
            f"Using AdaLN LoRA Flag:  {use_adaln_lora}. We enable bias if no AdaLN LoRA for backward compatibility."
        )
        self.in_dim = in_features
        self.out_dim = out_features
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=False)

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.in_dim)
        torch.nn.init.trunc_normal_(self.linear_1.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self.out_dim)
        torch.nn.init.trunc_normal_(self.linear_2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)

        if self.use_adaln_lora:
            adaln_lora_B_T_3D = emb
            emb_B_T_D = sample
        else:
            emb_B_T_D = emb
            adaln_lora_B_T_3D = None

        return emb_B_T_D, adaln_lora_B_T_3D


class FourierFeatures(nn.Module):
    """
    Implements a layer that generates Fourier features from input tensors, based on randomly sampled
    frequencies and phases. This can help in learning high-frequency functions in low-dimensional problems.

    [B] -> [B, D]

    Parameters:
        num_channels (int): The number of Fourier features to generate.
        bandwidth (float, optional): The scaling factor for the frequency of the Fourier features. Defaults to 1.
        normalize (bool, optional): If set to True, the outputs are scaled by sqrt(2), usually to normalize
                                    the variance of the features. Defaults to False.

    Example:
        >>> layer = FourierFeatures(num_channels=256, bandwidth=0.5, normalize=True)
        >>> x = torch.randn(10, 256)  # Example input tensor
        >>> output = layer(x)
        >>> print(output.shape)  # Expected shape: (10, 256)
    """

    def __init__(self, num_channels, bandwidth=1, normalize=False):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * bandwidth * torch.randn(num_channels), persistent=True)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels), persistent=True)
        self.gain = np.sqrt(2) if normalize else 1
        self.bandwidth = bandwidth
        self.num_channels = num_channels

        self.reset_parameters()

    def reset_parameters(self) -> None:
        generator = torch.Generator()
        generator.manual_seed(0)
        self.freqs = (
            2 * np.pi * self.bandwidth * torch.randn(self.num_channels, generator=generator).to(self.freqs.device)
        )
        self.phases = 2 * np.pi * torch.rand(self.num_channels, generator=generator).to(self.freqs.device)

    def forward(self, x, gain: float = 1.0):
        """
        Apply the Fourier feature transformation to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            gain (float, optional): An additional gain factor applied during the forward pass. Defaults to 1.

        Returns:
            torch.Tensor: The transformed tensor, with Fourier features applied.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32).ger(self.freqs.to(torch.float32)).add(self.phases.to(torch.float32))
        x = x.cos().mul(self.gain * gain).to(in_dtype)
        return x


class PatchEmbed(nn.Module):
    """
    PatchEmbed is a module for embedding patches from an input tensor by applying either 3D or 2D convolutional layers,
    depending on the . This module can process inputs with temporal (video) and spatial (image) dimensions,
    making it suitable for video and image processing tasks. It supports dividing the input into patches
    and embedding each patch into a vector of size `out_channels`.

    Parameters:
    - spatial_patch_size (int): The size of each spatial patch.
    - temporal_patch_size (int): The size of each temporal patch.
    - in_channels (int): Number of input channels. Default: 3.
    - out_channels (int): The dimension of the embedding vector for each patch. Default: 768.
    - bias (bool): If True, adds a learnable bias to the output of the convolutional layers. Default: True.
    """

    def __init__(
        self,
        spatial_patch_size,
        temporal_patch_size,
        in_channels=3,
        out_channels=768,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, out_channels, bias=False
            ),
        )
        self.dim = in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.proj[1].weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        """
        Forward pass of the PatchEmbed module.

        Parameters:
        - x (torch.Tensor): The input tensor of shape (B, C, T, H, W) where
            B is the batch size,
            C is the number of channels,
            T is the temporal dimension,
            H is the height, and
            W is the width of the input.

        Returns:
        - torch.Tensor: The embedded patches as a tensor, with shape b t h w c.
        """
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert (
            H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0
        ), f"H,W {(H,W)} should be divisible by spatial_patch_size {self.spatial_patch_size}"
        assert T % self.temporal_patch_size == 0
        x = self.proj(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of video DiT.
    """

    def __init__(
        self,
        hidden_size,
        spatial_patch_size,
        temporal_patch_size,
        out_channels,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False)
            )

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.trunc_normal_(self.linear.weight, std=std, a=-3 * std, b=3 * std)
        if self.use_adaln_lora:
            torch.nn.init.trunc_normal_(self.adaln_modulation[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation[1].weight)

        self.layer_norm.reset_parameters()

    def forward(
        self,
        # x_BT_HW_D,
        x_B_T_H_W_D,
        emb_B_T_D,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
    ):
        if self.use_adaln_lora:
            assert adaln_lora_B_T_3D is not None
            shift_B_T_D, scale_B_T_D = (
                self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
            ).chunk(2, dim=-1)
        else:
            shift_B_T_D, scale_B_T_D = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)

        shift_B_T_1_1_D, scale_B_T_1_1_D = rearrange(shift_B_T_D, "b t d -> b t 1 1 d"), rearrange(
            scale_B_T_D, "b t d -> b t 1 1 d"
        )

        def _fn(_x_B_T_H_W_D, _norm_layer, _scale_B_T_1_1_D, _shift_B_T_1_1_D):
            return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

        x_B_T_H_W_D = checkpoint(
            _fn, x_B_T_H_W_D, self.layer_norm, scale_B_T_1_1_D, shift_B_T_1_1_D, use_reentrant=False
        )
        x_B_T_H_W_O = self.linear(
            x_B_T_H_W_D
        )  # O = spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels
        return x_B_T_H_W_O


class Block(nn.Module):
    """
    A transformer block that combines self-attention, cross-attention and MLP layers with AdaLN modulation.
    Each component (self-attention, cross-attention, MLP) has its own layer normalization and AdaLN modulation.

    Parameters:
        x_dim (int): Dimension of input features
        context_dim (int): Dimension of context features for cross-attention
        num_heads (int): Number of attention heads
        mlp_ratio (float): Multiplier for MLP hidden dimension. Default: 4.0
        use_adaln_lora (bool): Whether to use AdaLN-LoRA modulation. Default: False
        adaln_lora_dim (int): Hidden dimension for AdaLN-LoRA layers. Default: 256

    The block applies the following sequence:
    1. Self-attention with AdaLN modulation
    2. Cross-attention with AdaLN modulation
    3. MLP with AdaLN modulation

    Each component uses skip connections and layer normalization.
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(x_dim, None, num_heads, x_dim // num_heads, qkv_format="bshd")

        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(x_dim, context_dim, num_heads, x_dim // num_heads, qkv_format="bshd")

        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        self.use_adaln_lora = use_adaln_lora
        if self.use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

    def reset_parameters(self) -> None:
        self.layer_norm_self_attn.reset_parameters()
        self.layer_norm_cross_attn.reset_parameters()
        self.layer_norm_mlp.reset_parameters()

        if self.use_adaln_lora:
            std = 1.0 / math.sqrt(self.x_dim)
            torch.nn.init.trunc_normal_(self.adaln_modulation_self_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_cross_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_mlp[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[1].weight)

    def init_weights(self) -> None:
        self.reset_parameters()
        self.self_attn.init_weights()
        self.cross_attn.init_weights()
        self.mlp.init_weights()

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        if self.use_adaln_lora:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
        else:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = self.adaln_modulation_self_attn(
                emb_B_T_D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = self.adaln_modulation_cross_attn(
                emb_B_T_D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        # Reshape tensors from (B, T, D) to (B, T, 1, 1, D) for broadcasting
        shift_self_attn_B_T_1_1_D = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d")
        scale_self_attn_B_T_1_1_D = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d")
        gate_self_attn_B_T_1_1_D = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d")

        shift_cross_attn_B_T_1_1_D = rearrange(shift_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        scale_cross_attn_B_T_1_1_D = rearrange(scale_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        gate_cross_attn_B_T_1_1_D = rearrange(gate_cross_attn_B_T_D, "b t d -> b t 1 1 d")

        shift_mlp_B_T_1_1_D = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d")
        scale_mlp_B_T_1_1_D = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d")
        gate_mlp_B_T_1_1_D = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d")

        B, T, H, W, D = x_B_T_H_W_D.shape

        def _fn(_x_B_T_H_W_D, _norm_layer, _scale_B_T_1_1_D, _shift_B_T_1_1_D):
            return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

        normalized_x_B_T_H_W_D = checkpoint(
            _fn,
            x_B_T_H_W_D,
            self.layer_norm_self_attn,
            scale_self_attn_B_T_1_1_D,
            shift_self_attn_B_T_1_1_D,
            use_reentrant=False,
        )
        result_B_T_H_W_D = rearrange(
            self.self_attn(
                # normalized_x_B_T_HW_D,
                rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                None,
                rope_emb=rope_emb_L_1_1_D,
                preserve_rng_state=True,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result_B_T_H_W_D

        def _x_fn(
            _x_B_T_H_W_D,
            layer_norm_cross_attn,
            _scale_cross_attn_B_T_1_1_D,
            _shift_cross_attn_B_T_1_1_D,
            _gate_cross_attn_B_T_1_1_D,
        ):
            _normalized_x_B_T_H_W_D = _fn(
                _x_B_T_H_W_D, layer_norm_cross_attn, _scale_cross_attn_B_T_1_1_D, _shift_cross_attn_B_T_1_1_D
            )
            _result_B_T_H_W_D = rearrange(
                self.cross_attn(
                    rearrange(_normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                    crossattn_emb,
                    rope_emb=rope_emb_L_1_1_D,
                    preserve_rng_state=True,
                ),
                "b (t h w) d -> b t h w d",
                t=T,
                h=H,
                w=W,
            )
            # _x_B_T_H_W_D = _x_B_T_H_W_D + _gate_cross_attn_B_T_1_1_D * _result_B_T_H_W_D
            return _result_B_T_H_W_D

        x_B_T_H_W_D = (
            checkpoint(
                _x_fn,
                x_B_T_H_W_D,
                self.layer_norm_cross_attn,
                scale_cross_attn_B_T_1_1_D,
                shift_cross_attn_B_T_1_1_D,
                gate_cross_attn_B_T_1_1_D,
                use_reentrant=False,
            )
            * gate_cross_attn_B_T_1_1_D
            + x_B_T_H_W_D
        )

        normalized_x_B_T_H_W_D = checkpoint(
            _fn, x_B_T_H_W_D, self.layer_norm_mlp, scale_mlp_B_T_1_1_D, shift_mlp_B_T_1_1_D, use_reentrant=False
        )
        result_B_T_H_W_D = self.mlp(normalized_x_B_T_H_W_D, preserve_rng_state=True)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_B_T_1_1_D * result_B_T_H_W_D
        return x_B_T_H_W_D


class GeneralDITTransformerBlock:
    pass


class SDXLTimestepEmbedding:
    pass


class SDXLTimesteps:
    pass
