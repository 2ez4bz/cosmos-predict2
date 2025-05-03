<p align="center">
    <img src="assets/nvidia-cosmos-header.png" alt="NVIDIA Cosmos Header">
</p>

### [Product Website](https://www.nvidia.com/en-us/ai/cosmos/) | [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-predict2-68028efc052239369a0f2959) | [Paper](https://arxiv.org/abs/2501.03575) | [Paper Website](https://research.nvidia.com/labs/dir/cosmos-predict1)

Cosmos-Predict2 is a key branch of Cosmos World Foundation Models (WFMs) specialized for future state prediction, often referred to as world models. The three main branches of Cosmos WFMs are [cosmos-predict](https://github.com/nvidia-cosmos/cosmos-predict2), [cosmos-transfer](https://github.com/nvidia-cosmos/cosmos-transfer1), and [cosmos-reason](https://github.com/nvidia-cosmos/cosmos-reason1). We visualize the architecture of Cosmos-Predict2 in the following figure.

<p align="center">
    <img src="assets/cosmos_predict_diagram.png" alt="Cosmos-Predict Architecture Diagram">
</p>

## Key Features
Cosmos-Predict2 includes the following:

- **Diffusion-based world foundation models** for Text2Image and Video2World generation, where a user can generate visual simulation based on text prompts or video prompts.

## News
- [2025/05] Initial release of Predict2! This is a major upgrade from Predict1. Please try it out and give us feedback. 

## Documentation
See below for quickstart installation and usage examples. 

### Install
Clone the `cosmos-predict2` source code
```bash
git clone git@github.com:nvidia-cosmos/cosmos-predict2.git
cd cosmos-predict2
```

> ℹ️ Cosmos runs only on Linux systems. We have tested the installation with Ubuntu 24.04, 22.04, and 20.04.
> ℹ️ Cosmos requires the Python version to be `3.10.x`. Please also make sure you have `conda` installed ([instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)).

The below commands creates the `cosmos-predict2` conda environment and installs the dependencies for inference:
```bash
# Create the cosmos-predict2 conda environment.
conda env create --file cosmos-predict2.yaml
# Activate the cosmos-predict2 conda environment.
conda activate cosmos-predict2
# Install the dependencies.
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
# Patch Transformer engine linking issues in conda environments.
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10
# Install Transformer engine.
pip install transformer-engine[pytorch]==1.12.0
```

You can test the environment setup for inference with
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/test_environment.py
```

Upcoming: we are working on easier installations options including a packaged Docker image. If you have questions of feedback, please file an issue and we will follow up!

### Download checkpoints
1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token (if you haven't done so already). Set the access token to `Read` permission (default is `Fine-grained`).

2. Log in to Hugging Face with the access token:
   ```bash
   huggingface-cli login
   ```
3. Accept the [LlamaGuard-7b terms](https://huggingface.co/meta-llama/LlamaGuard-7b)

4. Download the Cosmos model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-predict2-68028efc052239369a0f2959):
   ```bash
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_checkpoints.py --model_sizes 2B 14B --model_types Text2Image --checkpoint_dir checkpoints
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_checkpoints.py --model_sizes 2B 14B --model_types Video2World --checkpoint_dir checkpoints

### Inference with pre-trained Cosmos-Predict2 models
* [Inference with diffusion-based Text2Image models](/examples/inference_diffusion_text2image.md)
* [Inference with diffusion-based Video2World models](/examples/inference_diffusion_video2world.md)

## Models

Cosmos-Predict2 include the following models

* [Cosmos-Predict2-2B-Text2Image](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image): Text to image generation
* [Cosmos-Predict2-14B-Text2Image](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Text2Image): Text to image generation
* [Cosmos-Predict2-2B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World): Video + Text based future visual world generation
* [Cosmos-Predict2-14B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Video2World): Video + Text based future visual world generation

## Contribute

We thrive on community collaboration! [NVIDIA-Cosmos](https://github.com/nvidia-cosmos/) wouldn’t be where it is without contributions from developers like you. Check out our [Contributing Guide](CONTRIBUTING.md) to get started, and share your feedback through issues.

Big thanks 🙏 to everyone helping us push the boundaries of open-source physical AI!
<!-- ------------------------------ -->

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license (such as exemption of guardrail), please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
