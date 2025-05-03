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

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download

# from scripts.download_guardrail_checkpoints import download_guardrail_checkpoints


def parse_args():
    parser = argparse.ArgumentParser(description="Download NVIDIA Cosmos Predict1 diffusion models from Hugging Face")
    parser.add_argument(
        "--model_sizes",
        nargs="*",
        default=[
            "2B",
            "14B",
        ],  # Download all by default
        choices=["2B", "14B"],
        help="Which model sizes to download. Possible values: 7B, 14B",
    )
    parser.add_argument(
        "--model_types",
        nargs="*",
        default=[
            "Text2Image",
            "Video2World",
        ],  # Download all by default
        choices=["Text2Image", "Video2World"],
        help="Which model types to download. Possible values: Text2Image, Video2World",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Directory to save the downloaded checkpoints."
    )
    args = parser.parse_args()
    return args


MD5_CHECKSUM_LOOKUP = {
    # text2image
    "Cosmos-Predict2-2B-Text2Image/model.pt": "0074ac726bc2351eb166e2e90e804237",
    "Cosmos-Predict2-14B-Text2Image/model.pt": "9193a424bd16f07e4e7f1afcb543efcb",
    # video2world
    "Cosmos-Predict2-2B-Video2World/model.pt": "54608e54f296688b9665dc93b2ac13a8",
    "Cosmos-Predict2-14B-Video2World/model.pt": "4f27b429cc960911192b3886b9d3e11d",
    # t5
    "google-t5/t5-11b/pytorch_model.bin": "f890878d8a162e0045a25196e27089a3",
    "google-t5/t5-11b/tf_model.h5": "e081fc8bd5de5a6a9540568241ab8973",
    # wan2.1
    "Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth": "854fcb755005951fa5b329799af6199f",
    # Qwen2.5
    "Qwen/Qwen2.5-VL-7B-Instruct/model-00001-of-00005.safetensors": "8644c8b51bc77b2e0c050ce5e7be30f7",
    "Qwen/Qwen2.5-VL-7B-Instruct/model-00002-of-00005.safetensors": "1220b9cc11ac55254d4165e285c7e2d9",
    "Qwen/Qwen2.5-VL-7B-Instruct/model-00003-of-00005.safetensors": "4ec40524f0a3e91ec66dc5729d46d118",
    "Qwen/Qwen2.5-VL-7B-Instruct/model-00004-of-00005.safetensors": "7b5797821c1bbb60aa3dd9a65c88eedc",
    "Qwen/Qwen2.5-VL-7B-Instruct/model-00005-of-00005.safetensors": "ab4fb1ef087d7df0bc0df603e017d3cd",
}


def get_md5_checksum(checkpoints_dir, model_name):
    print("---------------------")
    for key, value in MD5_CHECKSUM_LOOKUP.items():
        if key.startswith(model_name + "/"):
            print(f"Verifying checkpoint {key}...")
            file_path = checkpoints_dir.joinpath(key)
            # File must exist
            if not Path(file_path).exists():
                print(f"Checkpoint {key} does not exist.")
                return False
            # File must match give MD5 checksum
            with open(file_path, "rb") as f:
                file_md5 = hashlib.md5(f.read()).hexdigest()
            if file_md5 != value:
                print(f"MD5 checksum of checkpoint {key} does not match.")
                return False
    print(f"Model checkpoints for {model_name} exist with matched MD5 checksums.")
    return True


def main(args):
    ORG_NAME = "nvidia"

    # Mapping from size argument to Hugging Face repository name
    model_map = {
        "2B": "Cosmos-Predict2-2B",
        "14B": "Cosmos-Predict2-14B",
    }

    # Additional models that are always downloaded
    extra_models = [
        "Wan-AI/Wan2.1-T2V-1.3B",
        "google-t5/t5-11b",
    ]

    if "Video2World" in args.model_types:
        # prompt upsampler for video2world
        extra_models.append("Qwen/Qwen2.5-VL-7B-Instruct")

    # Create local checkpoints folder
    checkpoints_dir = Path(args.checkpoint_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    download_kwargs = dict(
        allow_patterns=["README.md", "model.pt", "mean_std.pt", "config.json", "*.jit", "guardrail/*"]
    )

    # Download the requested diffusion models
    for size in args.model_sizes:
        for model_type in args.model_types:
            suffix = f"-{model_type}"
            model_name = model_map[size] + suffix
            repo_id = f"{ORG_NAME}/{model_name}"
            local_dir = checkpoints_dir.joinpath(model_name)

            if not get_md5_checksum(checkpoints_dir, model_name):
                local_dir.mkdir(parents=True, exist_ok=True)
                print(f"Downloading {repo_id} to {local_dir}...")
                snapshot_download(
                    repo_id=repo_id, local_dir=str(local_dir), local_dir_use_symlinks=False, **download_kwargs
                )

    # Download the always-included models
    for model_name in extra_models:
        if model_name in ("google-t5/t5-11b", "Wan-AI/Wan2.1-T2V-1.3B", "Qwen/Qwen2.5-VL-7B-Instruct"):
            repo_id = model_name
        else:
            repo_id = f"{ORG_NAME}/{model_name}"
        local_dir = checkpoints_dir.joinpath(model_name)

        if not get_md5_checksum(checkpoints_dir, model_name):
            local_dir.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {repo_id} to {local_dir}...")
            if model_name == "Wan-AI/Wan2.1-T2V-1.3B":
                download_kwargs = dict(allow_patterns=["README.md", "Wan2.1_VAE.pth", "config.json"])
            else:
                download_kwargs = dict()
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                **download_kwargs,
            )

    # download_guardrail_checkpoints(args.checkpoint_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
