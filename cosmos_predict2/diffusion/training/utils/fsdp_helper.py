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

from __future__ import annotations

from torch.distributed.device_mesh import init_device_mesh

from cosmos_predict2.utils import distributed, log

def hsdp_device_mesh(replica_group_size=None, sharding_group_size=None, device=None):
    """
     Initializes a device mesh for use with Hybrid Sharding strategy in FSDP (HSDP) training.

    This function requires explicit sizes for replica and sharding groups to accommodate models
    whose GPU fit is unknown, providing flexibility in distributed training setups.

    Args:
        replica_group_size (int): The size of each replica group. Must be provided to ensure
            the model fits within the available resources.
        sharding_group_size (int): The size of each sharding group that the model can fit. Must be provided to
            ensure the correct distribution of model parameters.
        device (str, optional): The device to use (e.g., "cuda:0"). If None, defaults to "cuda"
            with the local rank as the device index.

    Returns:
        A device mesh object compatible with FSDP.

    Raises:
        ValueError: If replica_group_size or sharding_group_size are not provided, or if the
            world size is not evenly divisible by the sharding group size.
        RuntimeError: If a valid device mesh cannot be created.

    Usage:
        If your model fits on 4 GPUS, and you have 3 nodes of 8 GPUs, then:
        Sharding_Group_Size = 4
        Replica_Groups_Size = (24 total gpus, 4 per sharding group) = 6 Replica Groups
        >>> device_mesh = initialize_device_mesh(replica_group_size, sharding_group_size)
        >>> sharded_model = FSDP(model, device_mesh=device_mesh, ...)
    """

    # world_size = int(os.getenv("WORLD_SIZE", "1"))
    world_size = distributed.get_world_size()
    if sharding_group_size is None:
        sharding_group_size = min(world_size, 8)
    sharding_group_size = min(sharding_group_size, world_size)
    if replica_group_size is None:
        replica_group_size = world_size // sharding_group_size

    device = device or "cuda"

    if world_size % sharding_group_size != 0:
        raise ValueError(
            f"World size {world_size} is not evenly divisible by " f"sharding group size {sharding_group_size}."
        )

    if (world_size // sharding_group_size) % replica_group_size != 0:
        raise ValueError(
            f"The calculated number of replica groups is not evenly divisible by "
            f"replica_group_size {replica_group_size}."
        )

    device_mesh = init_device_mesh(
        device, (replica_group_size, sharding_group_size), mesh_dim_names=("replicate", "shard")
    )
    if device_mesh is None:
        raise RuntimeError("Failed to create a valid device mesh.")

    log.critical(
        f"Device mesh initialized with replica group size {replica_group_size} and sharding group size {sharding_group_size}"
    )

    return device_mesh
