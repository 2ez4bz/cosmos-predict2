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

import torch
from diffusers import EDMEulerScheduler


class RectifiedFlowScheduler(EDMEulerScheduler):
    def precondition_outputs(self, sample, model_output, sigma):
        t = sigma / (sigma + 1)
        c_skip = 1.0 - t
        c_out = -t
        denoised = c_skip * sample + c_out * model_output

        return denoised

    def precondition_noise(self, sigma):
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor([sigma])

        c_noise = sigma / (sigma + 1)

        return c_noise

    def precondition_inputs(self, sample, sigma):
        t = sigma / (sigma + 1)
        c_in = 1.0 - t
        scaled_sample = sample * c_in
        return scaled_sample
