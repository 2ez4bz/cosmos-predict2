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

from cosmos_predict2.utils.lazy_config import LazyDict

Cosmos_Predict2_14B_Text2World: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /net": "cosmos_predict2_net_14b"},
            {"override /conditioner": "add_fps_padding_mask"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="Text2World",
            name="Cosmos_Predict2_14B_Text2World",
        ),
        model=dict(
            latent_shape=[
                16,
                16,
                88,
                160,
            ],
        ),
    )
)


cs = ConfigStore.instance()

for _item in [
    Cosmos_Predict2_14B_Text2World,
]:
    cs.store(group="experiment", package="_global_", name=_item["job"]["name"], node=_item)
