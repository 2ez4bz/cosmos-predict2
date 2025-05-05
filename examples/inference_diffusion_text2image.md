## Inference with diffusion-based Text2Image models

### Environment setup

Please refer to the Inference section of [INSTALL.md](/INSTALL.md#inference) for instructions on environment setup.

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
   ```

<!-- ### GPU memory requirements

We report the maximum observed GPU memory usage during end-to-end inference. Additionally, we offer a series of model offloading strategies to help users manage GPU memory usage effectively.

For GPUs with limited memory, we recommend fully offloading all models. For higher-end GPUs, users can select the most suitable offloading strategy considering the numbers provided below.

| Offloading Strategy | Cosmos-Predict2-2B-Text2World | Cosmos-Predict2-14B-Text2World |
|-------------|---------|---------|
| Offload prompt upsampler | - GB | > 80.0 GB |
| Offload prompt upsampler & guardrails | - GB | - GB |
| Offload prompt upsampler & guardrails & T5 encoder | - GB | - GB |
| Offload prompt upsampler & guardrails & T5 encoder & tokenizer | - GB | - GB |
| Offload prompt upsampler & guardrails & T5 encoder & tokenizer & diffusion model | - GB | - GB |

The numbers may vary depending on system specs and are for reference only. -->

### Examples

There are two models available for diffusion world generation from text input: `Cosmos-Predict2-2B-Text2Image` and `Cosmos-Predict2-14B-Text2Image`.

The inference script is `cosmos_predict2/diffusion/inference/text2image.py`.
It requires the input argument `--prompt` (text input).
To see the complete list of available arguments, run
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict2/diffusion/inference/text2image.py --help
```

We will set prompts with environment variables first.
Please refer to example inputs in [assets/text2image/](/assets/text2image/).
Below prompt is from [assets/text2image/input0.txt](/assets/text2image/input0.txt).
```bash
PROMPT="A close-up shot captures a vibrant yellow scrubber vigorously working on a grimy plate, its bristles moving in circular motions to lift stubborn grease and food residue. The dish, once covered in remnants of a hearty meal, gradually reveals its original glossy surface. Suds form and bubble around the scrubber, creating a satisfying visual of cleanliness in progress. The sound of scrubbing fills the air, accompanied by the gentle clinking of the dish against the sink. As the scrubber continues its task, the dish transforms, gleaming under the bright kitchen lights, symbolizing the triumph of cleanliness over mess."

NEGATIVE_PROMPT="The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
```

#### Example 1: single generation on the 2B model
This is the basic example for running inference on the 2B model with a single prompt.
The output is saved to `outputs/text2image_2b.jpg` alongside the corresponding prompt at `outputs/text2image_2b.txt`.
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict2/diffusion/inference/text2image.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict2-2B-Text2Image \
    --offload_prompt_upsampler \
    --disable_prompt_upsampler \
    --disable_guardrail \
    --prompt "${PROMPT}" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --image_save_name text2image_2b
```

#### Example 2: single generation on the 14B model
<!-- We run inference on the 14B model with offloading flags enabled. This is suitable for low-memory GPUs. Model offloading is also required for the 14B model to avoid OOM. -->
This is the basic example for running inference on the 14B model with a single prompt.
The output is saved to `outputs/text2image_14b.jpg` alongside the corresponding prompt at `outputs/text2image_14b.txt`.
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict2/diffusion/inference/text2image.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict2-14B-Text2Image \
    --offload_prompt_upsampler \
    --disable_prompt_upsampler \
    --disable_guardrail \
    --prompt "${PROMPT}" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --image_save_name text2image_14b
```

<!-- #### Example 3: single generation with multi-GPU inference
This example runs parallelized inference on a single prompt using 8 GPUs.
```bash
NUM_GPUS=8
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_predict2/diffusion/inference/text2image.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict2-14B-Text2Image \
    --offload_prompt_upsampler \
    --disable_prompt_upsampler \
    --disable_guardrail \
    --prompt "${PROMPT}" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --image_save_name text2image_14b_8gpu



NUM_GPUS=8
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_predict2/diffusion/inference/text2image.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict2-2B-Text2Image \
    --offload_prompt_upsampler \
    --disable_prompt_upsampler \
    --disable_guardrail \
    --prompt "${PROMPT}" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --image_save_name text2image_2b_8gpu
``` -->

<!-- #### Example 4: batch generation
This example runs inference on a batch of prompts, provided through the `--batch_input_path` argument (path to a JSONL file).
The JSONL file should contain one prompt per line in the following format, where each line must contain a `prompt` field:
```json
{"prompt": "prompt1"}
{"prompt": "prompt2"}
```
Inference command:
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict2/diffusion/inference/text2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict2-2B-Text2World \
    --batch_input_path assets/diffusion/batch_inputs/text2world.jsonl \
    --offload_prompt_upsampler \
    --video_save_folder diffusion-text2world-2b-batch
``` -->
