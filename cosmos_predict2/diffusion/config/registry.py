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

from hydra.core.config_store import ConfigStore

from cosmos_predict2.diffusion.config.base.conditioner import (
    BaseVideoConditionerConfig,
    VideoConditionerFpsPaddingConfig,
    VideoConditionerFpsSizePaddingConfig,
    VideoConditionerFpsSizePaddingFrameRepeatConfig,
    VideoExtendConditionerConfig,
    VideoExtendConditionerFrameRepeatConfig,
)
from cosmos_predict2.diffusion.config.base.net import (
    FADITV2_14B_Config,
    FADITV2Config,
    COSMOS_PREDICT2_NET_2B_Config,
    COSMOS_PREDICT2_NET_14B_Config,
)
from cosmos_predict2.diffusion.config.base.tokenizer import (
    get_cosmos_diffusion_tokenizer_comp8x8x8,
    get_wan2pt1_tokenizer,
)


def register_net(cs):
    cs.store(
        group="net",
        package="model.net",
        name="faditv2_7b",
        node=FADITV2Config,
    )
    cs.store(
        group="net",
        package="model.net",
        name="faditv2_14b",
        node=FADITV2_14B_Config,
    )
    cs.store(
        group="net",
        package="model.net",
        name="cosmos_predict2_net_2b",
        node=COSMOS_PREDICT2_NET_2B_Config,
    )
    cs.store(
        group="net",
        package="model.net",
        name="cosmos_predict2_net_14b",
        node=COSMOS_PREDICT2_NET_14B_Config,
    )


def register_conditioner(cs):
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="basic",
        node=BaseVideoConditionerConfig,
    )
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="add_fps_image_size_padding_mask",
        node=VideoConditionerFpsSizePaddingConfig,
    )
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="add_fps_padding_mask",
        node=VideoConditionerFpsPaddingConfig,
    )
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="video_cond",
        node=VideoExtendConditionerConfig,
    )
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="add_fps_image_size_padding_mask_frame_repeat",
        node=VideoConditionerFpsSizePaddingFrameRepeatConfig,
    )
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="video_cond_frame_repeat",
        node=VideoExtendConditionerFrameRepeatConfig,
    )


def register_tokenizer(cs):
    cs.store(
        group="tokenizer",
        package="model.tokenizer",
        name="cosmos_diffusion_tokenizer_res720_comp8x8x8_t121_ver092624",
        node=get_cosmos_diffusion_tokenizer_comp8x8x8(resolution="720", chunk_duration=121),
    )

    cs.store(
        group="tokenizer",
        package="model.tokenizer",
        name="wan2pt1_tokenizer",
        node=get_wan2pt1_tokenizer(),
    )


def register_configs():
    cs = ConfigStore.instance()

    register_net(cs)
    register_conditioner(cs)
    register_tokenizer(cs)
