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

from typing import List

import attrs

from cosmos_predict2.diffusion.training.modules.edm_sde import EDMSDE
from cosmos_predict2.utils.lazy_config import LazyCall as L
from cosmos_predict2.utils.lazy_config import LazyDict
from cosmos_predict2.diffusion.model.model_v2w import ConditioningStrategy


@attrs.define(slots=False)
class DefaultModelConfig:
    tokenizer: LazyDict = None
    conditioner: LazyDict = None
    net: LazyDict = None
    sde: LazyDict = L(EDMSDE)(
        p_mean=0.0,
        p_std=1.0,
        sigma_max=80,
        sigma_min=0.0002,
    )
    sigma_data: float = 1.0
    precision: str = "bfloat16"
    input_data_key: str = "video"  # key to fetch input data from data_batch
    latent_shape: List[int] = [16, 24, 44, 80]  # 24 corresponig to 136 frames
    state_ch: int = 16  # for latent model, ref to the latent channel number
    state_t: int = 8  # for latent model, ref to the latent number of frames
    resize_online: bool = (
        False  # whether or not resize the video online; usecase: we load a long duration video and resize to fewer frames, simulate low fps video. If true, it use tokenizer and state_t to infer the expected length of the resized video.
    )
    scaling: str = "rectified_flow"


@attrs.define(slots=False)
class LatentDiffusionDecoderModelConfig(DefaultModelConfig):
    tokenizer_corruptor: LazyDict = None
    latent_corruptor: LazyDict = None
    pixel_corruptor: LazyDict = None
    diffusion_decoder_cond_sigma_low: float = None
    diffusion_decoder_cond_sigma_high: float = None
    diffusion_decoder_corrupt_prob: float = None
    condition_on_tokenizer_corruptor_token: bool = False


@attrs.define(slots=False)
class Vid2VidModelConfig(DefaultModelConfig):
    min_num_conditional_frames: int = 1  # Minimum number of latent conditional frames
    max_num_conditional_frames: int = 2  # Maximum number of latent conditional frames
    sigma_conditional: float = 0.0001  # Noise level used for conditional frames
    conditioning_strategy: str = str(ConditioningStrategy.FRAME_REPLACE)  # What strategy to use for conditioning
