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
from typing import Callable, Dict, Optional, Tuple

import torch
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from cosmos_predict2.diffusion.conditioner import BaseVideoCondition, DataType, VideoExtendCondition
from cosmos_predict2.diffusion.model.model_t2w import DiffusionT2WModel, broadcast_condition
from cosmos_predict2.diffusion.module.parallel import cat_outputs_cp, split_inputs_cp
from cosmos_predict2.diffusion.modules.denoiser_scaling import RectifiedFlowScaling
from cosmos_predict2.diffusion.modules.res_sampler import Sampler
from cosmos_predict2.diffusion.training.context_parallel import broadcast, broadcast_split_tensor
from cosmos_predict2.diffusion.types import DenoisePrediction
from cosmos_predict2.utils import log, misc
from cosmos_predict2.utils.lazy_config import instantiate as lazy_instantiate

""
IS_PREPROCESSED_KEY = "is_preprocessed"
NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"


class ConditioningStrategy(Enum):
    FRAME_REPLACE = "frame_replace"  # First few frames of the video are replaced with the conditional frames
    CHANNEL_CONCAT = "channel_concat"  # First few frames of the video are concatenated in the channel dimension


class DiffusionV2WModel(DiffusionT2WModel):
    def __init__(self, config):
        super().__init__(config)
        self.sde = lazy_instantiate(config.sde)
        self.sampler = Sampler()
        self.scaling = RectifiedFlowScaling(self.sigma_data)

        # import ipdb

        # ipdb.set_trace()

    def add_condition_video_indicator_and_video_input_mask(
        self, latent_state: torch.Tensor, condition: VideoExtendCondition, num_condition_t: Optional[int] = None
    ) -> VideoExtendCondition:
        """Adds conditioning masks to VideoExtendCondition object.

        Creates binary indicators and input masks for conditional video generation.

        Args:
            latent_state: Input latent tensor (B,C,T,H,W)
            condition: VideoExtendCondition object to update
            num_condition_t: Number of frames to condition on

        Returns:
            Updated VideoExtendCondition with added masks:
            - condition_video_indicator: Binary tensor marking condition regions
            - condition_video_input_mask: Input mask for network
            - gt_latent: Ground truth latent tensor
        """
        T = latent_state.shape[2]
        latent_dtype = latent_state.dtype
        condition_video_indicator = torch.zeros(1, 1, T, 1, 1, device=latent_state.device).type(
            latent_dtype
        )  # 1 for condition region

        # Only in inference to decide the condition region
        assert num_condition_t is not None, "num_condition_t should be provided"
        assert num_condition_t <= T, f"num_condition_t should be less than T, get {num_condition_t}, {T}"
        log.debug(
            f"condition_location first_n, num_condition_t {num_condition_t}, condition.video_cond_bool {condition.video_cond_bool}"
        )
        condition_video_indicator[:, :, :num_condition_t] += 1.0

        condition.gt_latent = latent_state
        condition.condition_video_indicator = condition_video_indicator

        B, C, T, H, W = latent_state.shape
        # Create additional input_mask channel, this will be concatenated to the input of the network
        # See design doc section (Implementation detail A.1 and A.2) for visualization
        ones_padding = torch.ones((B, 1, T, H, W), dtype=latent_state.dtype, device=latent_state.device)
        zeros_padding = torch.zeros((B, 1, T, H, W), dtype=latent_state.dtype, device=latent_state.device)
        assert condition.video_cond_bool is not None, "video_cond_bool should be set"

        # The input mask indicate whether the input is conditional region or not
        if condition.video_cond_bool:  # Condition one given video frames
            condition.condition_video_input_mask = (
                condition_video_indicator * ones_padding + (1 - condition_video_indicator) * zeros_padding
            )
        else:  # Unconditional case, use for cfg
            condition.condition_video_input_mask = zeros_padding

        return condition

    def generate_samples_from_batch(
        self,
        data_batch: dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: tuple | None = None,
        n_sample: int | None = 1,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        condition_latent: Optional[torch.Tensor] = None,
        num_condition_t: Optional[int] = None,
        condition_augment_sigma: float = None,
        add_input_frames_guidance: bool = False,
    ) -> Tensor:
        """Generates video samples conditioned on input frames.

        Args:
            data_batch: Input data dictionary
            guidance: Classifier-free guidance scale
            seed: Random seed for reproducibility
            state_shape: Shape of output tensor (defaults to model's state shape)
            n_sample: Number of samples to generate (defaults to batch size)
            is_negative_prompt: Whether to use negative prompting
            num_steps: Number of denoising steps
            condition_latent: Conditioning frames tensor (B,C,T,H,W)
            num_condition_t: Number of frames to condition on
            condition_augment_sigma: Noise level for condition augmentation
            add_input_frames_guidance: Whether to apply guidance to input frames

        Returns:
            Generated video samples tensor
        """
        assert condition_latent is not None, "condition_latent should be provided"
        condition, uncondition = self._get_conditions(
            data_batch, is_negative_prompt, condition_latent, num_condition_t, add_input_frames_guidance
        )

        self.scheduler.set_timesteps(num_steps)
        if n_sample is None:
            n_sample = condition_latent.shape[0]
        xt = torch.randn(size=(n_sample,) + tuple(state_shape), **self.tensor_kwargs) * self.scheduler.init_noise_sigma

        to_cp = self.net.is_context_parallel_enabled
        if to_cp:
            xt = split_inputs_cp(x=xt, seq_dim=2, cp_group=self.net.cp_group)

        for t in self.scheduler.timesteps:
            self.scheduler._init_step_index(t)
            sigma = self.scheduler.sigmas[self.scheduler.step_index].to(**self.tensor_kwargs)
            # Form new noise from latent
            xt = xt.to(**self.tensor_kwargs)
            new_xt, latent, indicator = self._augment_noise_with_latent(
                xt, sigma, condition, condition_augment_sigma=condition_augment_sigma, seed=seed
            )
            new_xt = new_xt.to(**self.tensor_kwargs)
            new_xt_scaled = self.scheduler.scale_model_input(new_xt, timestep=t)
            # Predict the noise residual
            t = t.to(**self.tensor_kwargs)
            ones_B = torch.ones(new_xt.size(0), device=new_xt.device, dtype=t.dtype)
            t = t * ones_B
            net_output_cond = self.net(x_B_C_T_H_W=new_xt, timesteps_B_T=t, **condition.to_dict())
            net_output_uncond = self.net(x_B_C_T_H_W=new_xt, timesteps_B_T=t, **uncondition.to_dict())
            net_output = net_output_cond + guidance * (net_output_cond - net_output_uncond)
            # Replace indicated output with latent
            latent_unscaled = self._reverse_precondition_output(latent, xt=new_xt, sigma=sigma)
            new_output = indicator * latent_unscaled + (1 - indicator) * net_output
            # Compute the previous noisy sample x_t -> x_t-1
            xt = self.scheduler.step(new_output, t, new_xt).prev_sample
        samples = xt

        if to_cp:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.net.cp_group)

        return samples

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor]) -> Tuple[Tensor, BaseVideoCondition]:
        self._normalize_video_databatch_inplace(data_batch)
        # self._augment_image_dim_inplace(data_batch)
        # is_image_batch = False

        # Latent state
        raw_state = data_batch[self.input_data_key]
        latent_state = self.encode(raw_state).contiguous().float()

        # Condition
        condition = self.conditioner(data_batch)
        condition = condition.edit_data_type(DataType.VIDEO)

        condition = condition.set_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=None,
        )
        return raw_state, latent_state, condition

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        """
        Normalizes video data in-place on a CUDA device to reduce data loading overhead.

        This function modifies the video data tensor within the provided data_batch dictionary
        in-place, scaling the uint8 data from the range [0, 255] to the normalized range [-1, 1].

        Warning:
            A warning is issued if the data has not been previously normalized.

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the video data under a specific key.
                This tensor is expected to be on a CUDA device and have dtype of torch.uint8.

        Side Effects:
            Modifies the 'input_data_key' tensor within the 'data_batch' dictionary in-place.

        Note:
            This operation is performed directly on the CUDA device to avoid the overhead associated
            with moving data to/from the GPU. Ensure that the tensor is already on the appropriate device
            and has the correct dtype (torch.uint8) to avoid unexpected behaviors.
        """
        input_key = self.input_data_key if input_key is None else input_key
        # only handle video batch
        if input_key in data_batch:
            # Check if the data has already been normalized and avoid re-normalizing
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                assert torch.is_floating_point(data_batch[input_key]), "Video data is not in float format."
                assert torch.all(
                    (data_batch[input_key] >= -1.0001) & (data_batch[input_key] <= 1.0001)
                ), f"Video data is not in the range [-1, 1]. get data range [{data_batch[input_key].min()}, {data_batch[input_key].max()}]"
            else:
                assert data_batch[input_key].dtype == torch.uint8, "Video data is not in uint8 format."
                data_batch[input_key] = data_batch[input_key].to(**self.tensor_kwargs) / 127.5 - 1.0
                data_batch[IS_PREPROCESSED_KEY] = True

            if self.config.resize_online:
                expected_length = self.tokenizer.get_pixel_num_frames(self.config.state_t)
                if data_batch[input_key].shape[2] != expected_length:
                    H, W = data_batch[input_key].shape[3:]
                    data_batch[input_key] = torch.nn.functional.interpolate(
                        data_batch[input_key], size=(expected_length, H, W), mode="trilinear", align_corners=False
                    )

    @staticmethod
    def get_context_parallel_group():
        if parallel_state.is_initialized():
            return parallel_state.get_context_parallel_group()
        return None

    def broadcast_split_for_model_parallelsim(self, x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T):
        """
        Broadcast and split the input data and condition for model parallelism.
        Currently, we only support context parallelism.
        """
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if condition.is_video and cp_size > 1:
            x0_B_C_T_H_W = broadcast_split_tensor(x0_B_C_T_H_W, seq_dim=2, process_group=cp_group)
            epsilon_B_C_T_H_W = broadcast_split_tensor(epsilon_B_C_T_H_W, seq_dim=2, process_group=cp_group)
            if sigma_B_T is not None:
                assert sigma_B_T.ndim == 2, "sigma_B_T should be 2D tensor"
                if sigma_B_T.shape[-1] == 1:  # single sigma is shared across all frames
                    sigma_B_T = broadcast(sigma_B_T, cp_group)
                else:  # different sigma for each frame
                    sigma_B_T = broadcast_split_tensor(sigma_B_T, seq_dim=1, process_group=cp_group)
            if condition is not None:
                condition = condition.broadcast(cp_group)
            self.net.enable_context_parallel(cp_group)
        else:
            self.net.disable_context_parallel()

        return x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T

    def denoise(self, xt_B_C_T_H_W: torch.Tensor, sigma: torch.Tensor, condition):
        """
        Performs denoising on the input noise data, noise level, and condition

        Args:
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (T2VCondition): conditional information, generated from self.conditioner

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred).
        """

        if sigma.ndim == 1:
            sigma_B_T = rearrange(sigma, "b -> b 1")
        elif sigma.ndim == 2:
            sigma_B_T = sigma
        else:
            raise ValueError(f"sigma shape {sigma.shape} is not supported")

        sigma_B_1_T_1_1 = rearrange(sigma_B_T, "b t -> b 1 t 1 1")
        # get precondition for the network
        c_skip_B_1_T_1_1, c_out_B_1_T_1_1, c_in_B_1_T_1_1, c_noise_B_1_T_1_1 = self.scaling(sigma=sigma_B_1_T_1_1)

        net_state_in_B_C_T_H_W = xt_B_C_T_H_W * c_in_B_1_T_1_1

        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(net_state_in_B_C_T_H_W) / self.config.sigma_data
            if not condition.use_video_condition:
                # When using random dropout, we zero out the ground truth frames
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                net_state_in_B_C_T_H_W
            )

            if self.config.conditioning_strategy == str(ConditioningStrategy.FRAME_REPLACE):
                # In case of frame replacement strategy, replace the first few frames of the video with the conditional frames
                # Update the c_noise as the conditional frames are clean and have very low noise

                # Make the first few frames of x_t be the ground truth frames
                net_state_in_B_C_T_H_W = (
                    condition_state_in_B_C_T_H_W * condition_video_mask
                    + net_state_in_B_C_T_H_W * (1 - condition_video_mask)
                )

                # Adjust c_noise for the conditional frames
                sigma_cond_B_1_T_1_1 = torch.ones_like(sigma_B_1_T_1_1) * self.config.sigma_conditional
                _, _, _, c_noise_cond_B_1_T_1_1 = self.scaling(sigma=sigma_cond_B_1_T_1_1)
                condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
                c_noise_B_1_T_1_1 = c_noise_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + c_noise_B_1_T_1_1 * (
                    1 - condition_video_mask_B_1_T_1_1
                )
            elif self.config.conditioning_strategy == str(ConditioningStrategy.CHANNEL_CONCAT):
                # In case of channel concatenation strategy, concatenate the conditional frames in the channel dimension
                condition_state_in_masked_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask
                net_state_in_B_C_T_H_W = torch.cat([net_state_in_B_C_T_H_W, condition_state_in_masked_B_C_T_H_W], dim=1)

        else:
            # In case of image batch, simply concatenate the 0 frames when channel concat strategy is used
            if self.config.conditioning_strategy == str(ConditioningStrategy.CHANNEL_CONCAT):
                net_state_in_B_C_T_H_W = torch.cat(
                    [net_state_in_B_C_T_H_W, torch.zeros_like(net_state_in_B_C_T_H_W)], dim=1
                )

        # forward pass through the network
        net_output_B_C_T_H_W = self.net(
            x_B_C_T_H_W=net_state_in_B_C_T_H_W.to(
                **self.tensor_kwargs
            ),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            timesteps_B_T=c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(
                **self.tensor_kwargs
            ),  # Eq. 7 of https://arxiv.org/pdf/2206.00364.pdf
            **condition.to_dict(),
        ).float()

        # import ipdb

        # ipdb.set_trace()
        x0_pred_B_C_T_H_W = c_skip_B_1_T_1_1 * xt_B_C_T_H_W + c_out_B_1_T_1_1 * net_output_B_C_T_H_W
        if condition.is_video:
            # Set the first few frames to the ground truth frames. This will ensure that the loss is not computed for the first few frames.
            x0_pred_B_C_T_H_W = condition.gt_frames.type_as(
                x0_pred_B_C_T_H_W
            ) * condition_video_mask + x0_pred_B_C_T_H_W * (1 - condition_video_mask)

        # get noise prediction based on sde
        eps_pred_B_C_T_H_W = (xt_B_C_T_H_W - x0_pred_B_C_T_H_W) / sigma_B_1_T_1_1

        return DenoisePrediction(x0_pred_B_C_T_H_W, eps_pred_B_C_T_H_W, None)

    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """

        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            num_conditional_frames = 1

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        # is_image_batch = self.is_image_batch(data_batch)
        is_image_batch = False
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, x0, _ = self.get_data_and_condition(data_batch)

        # override condition with inference mode; num_conditional_frames used Here!
        condition = condition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
        )
        uncondition = uncondition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
        )
        condition = condition.edit_for_inference(is_cfg_conditional=True)
        uncondition = uncondition.edit_for_inference(is_cfg_conditional=False)
        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        if parallel_state.is_initialized():
            pass
        else:
            assert (
                not self.net.is_context_parallel_enabled
            ), "parallel_state is not initialized, context parallel should be turned off."

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_x0 = self.denoise(noise_x, sigma, condition).x0
            uncond_x0 = self.denoise(noise_x, sigma, uncondition).x0
            raw_x0 = cond_x0 + guidance * (cond_x0 - uncond_x0)
            # import ipdb

            # ipdb.set_trace()
            if "guided_image" in data_batch:
                # replacement trick that enables inpainting with base model
                assert "guided_mask" in data_batch, "guided_mask should be in data_batch if guided_image is present"
                guide_image = data_batch["guided_image"]
                guide_mask = data_batch["guided_mask"]
                raw_x0 = guide_mask * guide_image + (1 - guide_mask) * raw_x0
            return raw_x0

        return x0_fn

    def generate_samples_from_batch_i4(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        solver_option: str = "2ab",
        x_sigma_max: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.
            guidance (float): guidance weights
            seed (int): random seed
            state_shape (tuple): shape of the state, default to data batch if not provided
            n_sample (int): number of samples to generate
            is_negative_prompt (bool): use negative prompt t5 in uncondition if true
            num_steps (int): number of steps for the diffusion process
            solver_option (str): differential equation solver option, default to "2ab"~(mulitstep solver)
        """
        self._normalize_video_databatch_inplace(data_batch)
        input_key = "video"
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        x0_fn = self.get_x0_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)

        if x_sigma_max is None:
            x_sigma_max = (
                misc.arch_invariant_rand(
                    (n_sample,) + tuple(state_shape),
                    torch.float32,
                    self.tensor_kwargs["device"],
                    seed,
                )
                * self.sde.sigma_max
            )

        if self.net.is_context_parallel_enabled:
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.get_context_parallel_group())

        # import ipdb

        # ipdb.set_trace()

        samples = self.sampler(
            x0_fn, x_sigma_max, num_steps=num_steps, sigma_max=self.sde.sigma_max, solver_option=solver_option
        )
        if self.net.is_context_parallel_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.get_context_parallel_group())

        return samples

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.decode(latent / self.sigma_data)

    def _get_conditions(
        self,
        data_batch: dict,
        is_negative_prompt: bool = False,
        condition_latent: Optional[torch.Tensor] = None,
        num_condition_t: Optional[int] = None,
        add_input_frames_guidance: bool = False,
    ):
        """Get the conditions for the model.

        Args:
            data_batch: Input data dictionary
            is_negative_prompt: Whether to use negative prompting
            condition_latent: Conditioning frames tensor (B,C,T,H,W)
            num_condition_t: Number of frames to condition on
            add_input_frames_guidance: Whether to apply guidance to input frames

        Returns:
            condition: Input conditions
            uncondition: Conditions removed/reduced to minimum (unconditioned)
        """
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        condition.video_cond_bool = True
        condition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, condition, num_condition_t
        )
        uncondition.video_cond_bool = False if add_input_frames_guidance else True
        uncondition = self.add_condition_video_indicator_and_video_input_mask(
            condition_latent, uncondition, num_condition_t
        )
        assert condition.gt_latent.allclose(uncondition.gt_latent)

        # For inference, check if parallel_state is initialized
        to_cp = self.net.is_context_parallel_enabled
        if parallel_state.is_initialized():
            condition = broadcast_condition(condition, to_tp=False, to_cp=to_cp)
            uncondition = broadcast_condition(uncondition, to_tp=False, to_cp=to_cp)

        return condition, uncondition

    def _augment_noise_with_latent(
        self,
        xt: Tensor,
        sigma: Tensor,
        condition: VideoExtendCondition,
        condition_augment_sigma: float = 0.001,
        seed: int = 1,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Augments the conditional frames with noise during inference.

        Args:
            xt (Tensor): noise
            sigma (Tensor): noise level for the generation region
            condition (VideoExtendCondition): condition object
                condition_video_indicator: binary tensor indicating the region is condition(value=1) or generation(value=0). Bx1xTx1x1 tensor.
                condition_video_input_mask: input mask for the network input, indicating the condition region. B,1,T,H,W tensor. will be concat with the input for the network.
            condition_augment_sigma (float): sigma for condition video augmentation in inference
            seed (int): random seed for reproducibility
        Returns:
            new_xt (Tensor): new latent-augmented noise tensor in shape B,C,T,H,W
            latent (Tensor): ground-truth latent tensor in shape B,C,T,H,W
            indicator (Tensor): ground-truth latent binary indicator tensor in shape B,C,T,H,W

        """
        # Augment the latent with different sigma value, and add the augment_sigma to the condition object if needed
        augment_sigma = condition_augment_sigma
        latent = condition.gt_latent
        indicator = condition.condition_video_indicator
        if augment_sigma >= sigma:
            indicator = torch.zeros_like(indicator)
        # Now apply the augment_sigma to the gt_latent
        noise = misc.arch_invariant_rand(
            latent.shape,
            torch.float32,
            self.tensor_kwargs["device"],
            seed,
        )
        augment_latent = latent + noise * augment_sigma
        augment_latent = self.scheduler.precondition_inputs(augment_latent, augment_sigma)
        augment_latent_unscaled = self._reverse_precondition_input(augment_latent, sigma)
        if self.net.is_context_parallel_enabled:
            latent = split_inputs_cp(condition.gt_latent, seq_dim=2, cp_group=self.net.cp_group)
            indicator = split_inputs_cp(indicator, seq_dim=2, cp_group=self.net.cp_group)
            augment_latent_unscaled = split_inputs_cp(augment_latent_unscaled, seq_dim=2, cp_group=self.net.cp_group)
        # Compose the model input with condition region (augment_latent) and generation region (noise_x)
        new_xt = indicator * augment_latent_unscaled + (1 - indicator) * xt
        return new_xt, latent, indicator

    def _reverse_precondition_input(self, xt: Tensor, sigma: Tensor) -> Tensor:
        c_in = 1 / ((sigma**2 + self.config.sigma_data**2) ** 0.5)
        xt_unscaled = xt / c_in
        return xt_unscaled

    def _reverse_precondition_output(self, latent: Tensor, xt: Tensor, sigma: Tensor) -> Tensor:
        sigma_data = self.scheduler.config.sigma_data
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        latent_unscaled = (latent - c_skip * xt) / c_out
        return latent_unscaled
