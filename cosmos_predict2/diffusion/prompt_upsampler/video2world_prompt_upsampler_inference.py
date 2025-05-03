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

"""
This demo script is used to run inference for Pixtral-12B.
Command:
    CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict2/diffusion/prompt_upsampler/video2world_prompt_upsampler_inference.py

"""

import argparse
import os

from cosmos_predict2.auxiliary.guardrail.common import presets as guardrail_presets

from cosmos_predict2.utils import log

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch


VL_EN_SYS_PROMPT =  \
    '''You are a prompt optimization specialist whose goal is to rewrite the user's input prompts into high-quality English prompts by referring to the details of the user's input images, making them more complete and expressive while maintaining the original meaning. You need to integrate the content of the user's photo with the input prompt for the rewrite, strictly adhering to the formatting of the examples provided.\n''' \
    '''Task Requirements:\n''' \
    '''1. For overly brief user inputs, reasonably infer and supplement details without changing the original meaning, making the image more complete and visually appealing;\n''' \
    '''2. Improve the characteristics of the main subject in the user's description (such as appearance, expression, quantity, ethnicity, posture, etc.), rendering style, spatial relationships, and camera angles;\n''' \
    '''3. The overall output should be in Chinese, retaining original text in quotes and book titles as well as important input information without rewriting them;\n''' \
    '''4. The prompt should match the user’s intent and provide a precise and detailed style description. If the user has not specified a style, you need to carefully analyze the style of the user's provided photo and use that as a reference for rewriting;\n''' \
    '''5. If the prompt is an ancient poem, classical Chinese elements should be emphasized in the generated prompt, avoiding references to Western, modern, or foreign scenes;\n''' \
    '''6. You need to emphasize movement information in the input and different camera angles;\n''' \
    '''7. Your output should convey natural movement attributes, incorporating natural actions related to the described subject category, using simple and direct verbs as much as possible;\n''' \
    '''8. You should reference the detailed information in the image, such as character actions, clothing, backgrounds, and emphasize the details in the photo;\n''' \
    '''9. Control the rewritten prompt to around 80-100 words.\n''' \
    '''10. No matter what language the user inputs, you must always output in English.\n''' \
    '''Example of the rewritten English prompt:\n''' \
    '''1. A Japanese fresh film-style photo of a young East Asian girl with double braids sitting by the boat. The girl wears a white square collar puff sleeve dress, decorated with pleats and buttons. She has fair skin, delicate features, and slightly melancholic eyes, staring directly at the camera. Her hair falls naturally, with bangs covering part of her forehead. She rests her hands on the boat, appearing natural and relaxed. The background features a blurred outdoor scene, with hints of blue sky, mountains, and some dry plants. The photo has a vintage film texture. A medium shot of a seated portrait.\n''' \
    '''2. An anime illustration in vibrant thick painting style of a white girl with cat ears holding a folder, showing a slightly dissatisfied expression. She has long dark purple hair and red eyes, wearing a dark gray skirt and a light gray top with a white waist tie and a name tag in bold Chinese characters that says "紫阳" (Ziyang). The background has a light yellow indoor tone, with faint outlines of some furniture visible. A pink halo hovers above her head, in a smooth Japanese cel-shading style. A close-up shot from a slightly elevated perspective.\n''' \
    '''3. CG game concept digital art featuring a huge crocodile with its mouth wide open, with trees and thorns growing on its back. The crocodile's skin is rough and grayish-white, resembling stone or wood texture. Its back is lush with trees, shrubs, and thorny protrusions. With its mouth agape, the crocodile reveals a pink tongue and sharp teeth. The background features a dusk sky with some distant trees, giving the overall scene a dark and cold atmosphere. A close-up from a low angle.\n''' \
    '''4. In the style of an American drama promotional poster, Walter White sits in a metal folding chair wearing a yellow protective suit, with the words "Breaking Bad" written in sans-serif English above him, surrounded by piles of dollar bills and blue plastic storage boxes. He wears glasses, staring forward, dressed in a yellow jumpsuit, with his hands resting on his knees, exuding a calm and confident demeanor. The background shows an abandoned, dim factory with light filtering through the windows. There’s a noticeable grainy texture. A medium shot with a straight-on close-up of the character.\n''' \
    '''Directly output the rewritten English text.'''

def create_vlm_prompt_upsampler(checkpoint_dir: str) -> tuple[AutoProcessor, Qwen2_5_VLForConditionalGeneration]:
    """Create a Qwen2.5 VL model for prompt upsampling.

    Args:
        checkpoint_dir (str): The checkpoint directory of the prompt upsampler.

    Returns:
        tuple[AutoProcessor, Qwen2_5_VLForConditionalGeneration]: The processor and model.
    """
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        checkpoint_dir,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cpu")
    return processor, model



def prepare_dialog(image_or_video_path: str, prompt: str) -> list[dict]:
    if image_or_video_path.endswith(".mp4"):
        type="video"
    elif image_or_video_path.endswith(".jpg") or image_or_video_path.endswith(".png"):
        type="image"
    else:
        raise ValueError(f"Unsupported file type: {image_or_video_path}")

    messages = [
        {
            "role": "system",
            "content": VL_EN_SYS_PROMPT,
        }, 
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": type,
                    type: image_or_video_path
                }
            ]
        }
    ]
    return messages

def run_chat_completion(processor: AutoProcessor, model: Qwen2_5_VLForConditionalGeneration, dialog: list[dict]) -> str:
    model = model.to("cuda")
    text = processor.apply_chat_template(
        dialog, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(dialog)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # Inference: Generation of the output
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    model = model.to("cpu")
    return output_text


def parse_args():
    parser = argparse.ArgumentParser(description="Run prompt upsampler inference")
    parser.add_argument("--image_or_video_path", type=str, default="assets/diffusion/video2world_input0.jpg")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Base directory containing model checkpoints"
    )
    parser.add_argument(
        "--prompt_upsampler_dir",
        type=str,
        default="Pixtral-12B",
        help="Prompt upsampler weights directory relative to checkpoint_dir",
    )
    return parser.parse_args()


def main(args):
    guardrail_runner = guardrail_presets.create_text_guardrail_runner(args.checkpoint_dir)

    processor, model = create_vlm_prompt_upsampler(os.path.join(args.checkpoint_dir, args.prompt_upsampler_dir))
    dialog = prepare_dialog(args.image_or_video_path, args.prompt)
    upsampled_prompt = run_chat_completion(
        processor,
        model,
        dialog
    )
    is_safe = guardrail_presets.run_text_guardrail(upsampled_prompt, guardrail_runner)
    if not is_safe:
        log.critical("Upsampled text prompt is not safe.")
        return

    log.info(f"Upsampled prompt: {upsampled_prompt}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
