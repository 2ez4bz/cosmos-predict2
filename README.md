<p align="center">
    <img src="assets/nvidia-cosmos-header.png" alt="NVIDIA Cosmos Header">
</p>

### [Product Website](https://www.nvidia.com/en-us/ai/cosmos/) | [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-predict1-67c9d1b97678dbf7669c89a7) | [Paper](https://arxiv.org/abs/2501.03575) | [Paper Website](https://research.nvidia.com/labs/dir/cosmos-predict1)

Cosmos-Predict2 is a key branch of Cosmos World Foundation Models (WFMs) specialized for future state prediction, often referred to as world models. The tree main branches of Cosmos WFMs are [cosmos-predict](https://github.com/nvidia-cosmos/cosmos-predict2), [cosmos-transfer](https://github.com/nvidia-cosmos/cosmos-transfer1), and [cosmos-reason](https://github.com/nvidia-cosmos/cosmos-reason1). We visualize the architecture of Cosmos-Predict2 in the following figure.

<p align="center">
    <img src="assets/predict1_diagram.png" alt="Cosmos-Predict1 Architecture Diagram">
</p>

## Key Features
Cosmos-Predict2 includes the following:

- **Diffusion-based world foundation models** for Text2World and Video2World generation, where a user can generate visual simulation based on text prompts and video prompts.
- **Post-training scripts** for helping Physical AI builders post-train pre-trained Cosmos-Predict2 for their applications.

## Documentation
See below for quickstart installation and usage examples. For comprehensive guidance on run inference and post-training, refer to our []

### Install
Please refer to [INSTALL.md](./INSTALL.md) for general instructions on environment setup.

<details open>
  <summary>Install</summary>

### Environment setup

Clone the `cosmos-predict2` source code
```bash
git clone git@github.com:nvidia-cosmos/cosmos-predict2.git
cd cosmos-predict2
```

Cosmos runs only on Linux systems. We have tested the installation with Ubuntu 24.04, 22.04, and 20.04.
Cosmos requires the Python version to be `3.10.x`. Please also make sure you have `conda` installed ([instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)).

#### Inference

The below commands creates the `cosmos-predict2` conda environment and installs the dependencies for inference:

#### Post-training

</details>

## Models

Cosmos-Predict2 include the following models

* [Cosmos-Predict2-2B-Text2World](): Text to visual world generation
* [Cosmos-Predict2-14B-Text2World](): Text to visual world generation
* [Cosmos-Predict2-2B-Video2World](): Video + Text based future visual world generation
* [Cosmos-Predict2-14B-Video2World](): Video + Text based future visual world generation

### Inference with pre-trained Cosmos-Predict2 models
* [Inference with diffusion-based Text2World models](/examples/inference_diffusion_text2world.md) 

## Contribute

We thrive on community collaboration! [NVIDIA-Cosmos](https://github.com/nvidia-cosmos/) wouldn’t be where it is without contributions from developers like you. Check out our Contributing Guide to get started, and share your feedback through issues.

Big thanks 🙏 to everyone helping us push the boundaries of open-source physical AI!
<!-- ------------------------------ -->

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license (such as exemption of guardrail), please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
