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

import copy
from typing import Dict
from hydra.core.config_store import ConfigStore

from cosmos_predict2.diffusion.training.conditioner import VideoExtendConditioner, VideoExtendConditionerV2
from cosmos_predict2.diffusion.config.base.conditioner import (
    FPSConfig,
    PaddingMaskConfig,
    TextConfig,
    UseVideoConditionConfig,
)
from cosmos_predict2.diffusion.training.config.base.optim import FusedAdamWConfig, LambdaLinearSchedulerConfig
from cosmos_predict2.diffusion.training.config.base.vae import get_wan2pt1_tokenizer
from cosmos_predict2.diffusion.training.networks.general_dit_lvg import VideoExtendGeneralDIT
from cosmos_predict2.utils.lazy_config import LazyCall as L
from cosmos_predict2.utils.lazy_config import LazyDict
from cosmos_predict2.diffusion.training.config.base.model import PowerEMAConfig
from cosmos_predict2.diffusion.checkpointers.dcp_checkpointer import CheckpointConfig, DistributedCheckpointer
from cosmos_predict2.diffusion.training.config.video2world.experiment import register_experiments

DCP_CHECKPOINTER: Dict[str, str] = L(DistributedCheckpointer)()

VideoPredictionConditioner: LazyDict = L(VideoExtendConditioner)(
    text=TextConfig(),
    fps=FPSConfig(),
    padding_mask=PaddingMaskConfig(),
    use_video_condition=UseVideoConditionConfig(),
)

VideoPredictionConditionerV2: LazyDict = L(VideoExtendConditionerV2)(
    text=TextConfig(),
    fps=FPSConfig(),
    padding_mask=PaddingMaskConfig(),
    use_video_condition=UseVideoConditionConfig(),
)


def register_conditioner(cs):
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="video_prediction_conditioner",
        node=VideoPredictionConditioner,
    )
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="video_prediction_conditioner_v2",
        node=VideoPredictionConditionerV2,
    )


def register_checkpoint_credential(cs):
    CHECKPOINT_LOCAL = CheckpointConfig(
        save_iter=1000,
        load_path="",
        load_training_state=False,
        strict_resume=True,
    )

    cs.store(group="checkpoint", package="checkpoint", name="local", node=CHECKPOINT_LOCAL)


def register_checkpointer(cs):
    cs.store(group="ckpt_klass", package="checkpoint.type", name="dcp", node=DCP_CHECKPOINTER)


COSMOS_PREDICT2_NET_V2W_2B_Config: LazyDict = L(VideoExtendGeneralDIT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    min_fps=1,
    max_fps=30,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=False,
)

COSMOS_PREDICT2_NET_V2W_14B_Config: LazyDict = L(VideoExtendGeneralDIT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=5120,
    num_blocks=36,
    num_heads=40,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    min_fps=1,
    max_fps=30,
    rope_h_extrapolation_ratio=2.0,
    rope_w_extrapolation_ratio=2.0,
    rope_t_extrapolation_ratio=20 / 24,
    extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=False,
)


def register_net(cs):
    cs.store(group="net", package="model.net", name="cosmos_predict2_net_v2w_2b", node=COSMOS_PREDICT2_NET_V2W_2B_Config)
    cs.store(group="net", package="model.net", name="cosmos_predict2_net_v2w_14b", node=COSMOS_PREDICT2_NET_V2W_14B_Config)


def register_vae(cs):
    cs.store(
        group="tokenizer",
        package="model.tokenizer",
        name="wan2pt1_tokenizer",
        node=get_wan2pt1_tokenizer(),
    )


def register_ema(cs):
    cs.store(group="ema", package="model.ema", name="power", node=PowerEMAConfig)


def register_optimizer(cs):
    cs.store(group="optimizer", package="optimizer", name="fusedadamw", node=FusedAdamWConfig)


def register_scheduler(cs):
    cs.store(group="scheduler", package="scheduler", name="lambdalinear", node=LambdaLinearSchedulerConfig)


def register_configs():
    cs = ConfigStore.instance()

    register_optimizer(cs)
    register_scheduler(cs)

    register_net(cs)
    register_conditioner(cs)
    register_vae(cs)

    register_ema(cs)

    register_checkpoint_credential(cs)
    register_checkpointer(cs)

    register_experiments(cs)