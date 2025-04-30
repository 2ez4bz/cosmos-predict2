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

from torch.distributed import ProcessGroup, all_gather, broadcast_object_list, get_process_group_ranks, get_world_size
from torch.distributed.utils import _verify_param_shape_across_processes
from cosmos_predict1.utils import distributed
import torch
from typing import Optional

def robust_broadcast(tensor: torch.Tensor, src: int, pg, is_check_shape: bool = False) -> torch.Tensor:
    """
    Perform a robust broadcast operation that works regardless of tensor shapes on different ranks.

    Args:
        tensor (torch.Tensor): The tensor to broadcast (on src rank) or receive (on other ranks).
        src (int): The source rank for the broadcast. Defaults to 0.

    Returns:
        torch.Tensor: The broadcasted tensor on all ranks.
    """
    # First, broadcast the shape of the tensor
    if distributed.get_rank() == src:
        shape = torch.tensor(tensor.shape).cuda()
    else:
        shape = torch.empty(tensor.dim(), dtype=torch.long).cuda()
    if is_check_shape:
        _verify_param_shape_across_processes(pg, [shape])
    torch.distributed.broadcast(shape, src, group=pg)

    # Resize the tensor on non-src ranks if necessary
    if distributed.get_rank() != src:
        tensor = tensor.new_empty(shape.tolist()).type_as(tensor)

    # Now broadcast the tensor data
    torch.distributed.broadcast(tensor, src, group=pg)

    return tensor

def broadcast(
    item: torch.Tensor | str | None, process_group: Optional[ProcessGroup] = None
) -> torch.Tensor | str | None:
    """
    Broadcast the item from the minimum rank in the specified group(s).
    """
    if process_group is None:
        return item

    min_rank = min(get_process_group_ranks(process_group))
    if isinstance(item, torch.Tensor):  # assume the device is cuda
        item = robust_broadcast(item, min_rank, process_group)
    elif item is not None:
        broadcastable_list = [item]
        broadcast_object_list(broadcastable_list, min_rank, group=process_group)
        item = broadcastable_list[0]
    return item