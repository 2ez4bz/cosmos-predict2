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

from enum import Enum
from typing import List

import attrs

from cosmos_predict2.diffusion.training.config.base.ema import PowerEMAConfig, EMAConfig
from cosmos_predict2.diffusion.training.modules.edm_sde import EDMSDE
from cosmos_predict2.utils.lazy_config import LazyCall as L
from cosmos_predict2.utils.lazy_config import LazyDict


@attrs.define(slots=False)
class DefaultModelConfig:
    """
    Config for [DiffusionModel][projects.cosmos.diffusion.v2.models.t2v_model.DiffusionModel].
    """

    tokenizer: LazyDict = None
    conditioner: LazyDict = None
    net: LazyDict = None
    ema: EMAConfig = PowerEMAConfig
    sde: LazyDict = L(EDMSDE)(
        p_mean=0.0,
        p_std=1.0,
        sigma_max=80,
        sigma_min=0.0002,
    )
    fsdp_shard_size: int = 1
    sigma_data: float = 1.0
    precision: str = "bfloat16"
    input_data_key: str = "video"  # key to fetch input data from data_batch
    input_image_key: str = "images"  # key to fetch input image from data_batch
    loss_reduce: str = "mean"
    loss_scale: float = 10.0
    use_torch_compile: bool = False
    adjust_video_noise: bool = True  # whether or not adjust video noise accroding to the video length

    state_ch: int = 16  # for latent model, ref to the latent channel number
    state_t: int = 8  # for latent model, ref to the latent number of frames
    resolution: str = "512"
    scaling: str = "rectified_flow"
    resize_online: bool = (
        False  # whether or not resize the video online; usecase: we load a long duration video and resize to fewer frames, simulate low fps video. If true, it use tokenizer and state_t to infer the expected length of the resized video.
    )

    def __post_init__(self):
        assert self.scaling == "rectified_flow"


class ConditioningStrategy(Enum):
    FRAME_REPLACE = "frame_replace"  # First few frames of the video are replaced with the conditional frames
    CHANNEL_CONCAT = "channel_concat"  # First few frames of the video are concatenated in the channel dimension


@attrs.define(slots=False)
class Vid2VidModelConfig(DefaultModelConfig):
    min_num_conditional_frames: int = 1  # Minimum number of latent conditional frames
    max_num_conditional_frames: int = 2  # Maximum number of latent conditional frames
    sigma_conditional: float = 0.0001  # Noise level used for conditional frames
    conditioning_strategy: str = str(ConditioningStrategy.FRAME_REPLACE)  # What strategy to use for conditioning
