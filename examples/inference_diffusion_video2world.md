## Inference with diffusion-based Video2World models

### Environment setup

Please refer to the Inference section of [INSTALL.md](/INSTALL.md#inference) for instructions on environment setup.

### Download checkpoints

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token (if you haven't done so already). Set the access token to `Read` permission (default is `Fine-grained`).

2. Log in to Hugging Face with the access token:
   ```bash
   huggingface-cli login
   ```
3. Accept the [Llama-Guard-3-8B terms](https://huggingface.co/meta-llama/Llama-Guard-3-8B)

4. Download the Cosmos model weights from [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-predict2-68028efc052239369a0f2959):
   ```bash
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_checkpoints.py --model_sizes 2B 14B --model_types Video2World --checkpoint_dir checkpoints
   ```

<!-- ### GPU memory requirements

We report the maximum observed GPU memory usage during end-to-end inference. Additionally, we offer a series of model offloading strategies to help users manage GPU memory usage effectively.

For GPUs with limited memory, we recommend fully offloading all models. For higher-end GPUs, users can select the most suitable offloading strategy considering the numbers provided below.

| Offloading Strategy                                                              | Cosmos-Predict2-2B-Video2World | Cosmos-Predict2-14B-Video2World |
|----------------------------------------------------------------------------------|---------|---------|
| Offload prompt upsampler                                                         | 76.5 GB | > 80.0 GB |
| Offload prompt upsampler & guardrails                                            | 59.9 GB | 73.3 GB |
| Offload prompt upsampler & guardrails & T5 encoder                               | 41.3 GB | 54.8 GB |
| Offload prompt upsampler & guardrails & T5 encoder & tokenizer                   | 41.1 GB | 54.5 GB |
| Offload prompt upsampler & guardrails & T5 encoder & tokenizer & diffusion model | 27.3 GB | 39.0 GB |

The numbers may vary depending on system specs and are for reference only. -->

### Examples

There are two models available for diffusion world generation from text and image/video input: `Cosmos-Predict2-2B-Video2World` and `Cosmos-Predict2-14B-Video2World`.

The inference script is `cosmos_predict2/diffusion/inference/video2world.py`.
It requires the input argument `--input_image_or_video_path` (image/video input); if the prompt upsampler is disabled, `--prompt` (text input) must also be provided.
To see the complete list of available arguments, run
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict2/diffusion/inference/video2world.py --help
```

We will set prompts with environment variables first.
Please refer to example inputs in [assets/video2world/](/assets/video2world/).
Below prompt is from [assets/video2world/input0.txt](/assets/video2world/input0.txt).
```bash
PROMPT="A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled “87D” facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene."

NEGATIVE_PROMPT="The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
```

#### Example 1: single generation on the 2B model
This is the basic example for running inference on the 2B model with a single image. 
The output is saved to `outputs/video2world_2b.mp4` alongside the corresponding prompt at `outputs/video2world_2b.txt`.

```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict2/diffusion/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --input_image_or_video_path assets/video2world/input0.jpg \
    --num_input_frames 1 \
    --diffusion_transformer_dir Cosmos-Predict2-2B-Video2World \
    --offload_prompt_upsampler \
    --disable_prompt_upsampler \
    --disable_guardrail \
    --prompt "${PROMPT}" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --height 432 --width 768 --num_video_frames 81 \
    --num_steps 35 \
    --video_save_name video2world_2b
```

We also support running with prompt upsampler that extends short prompts to add more details to help generating videos.
```bash
PROMPT="A nighttime city bus terminal."

CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict2/diffusion/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --prompt_upsampler_dir Qwen/Qwen2.5-VL-7B-Instruct \
    --input_image_or_video_path assets/video2world/input0.jpg \
    --num_input_frames 1 \
    --diffusion_transformer_dir Cosmos-Predict2-2B-Video2World \
    --offload_prompt_upsampler \
    --disable_guardrail \
    --prompt "${PROMPT}" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --height 432 --width 768 --num_video_frames 81 \
    --num_steps 35 \
    --video_save_name video2world_2b_prompt_upsampled
```

With the prompt upsampler enabled, the video is generated from the below upsampled prompt.
```bash
A nighttime city bus terminal illuminated by artificial lights, showcasing a bustling urban environment. Several buses, including a prominent yellow double-decker, are parked along the platform, their headlights and taillights casting a warm glow. The terminal is set against a backdrop of tall buildings, their windows reflecting the city lights. The sky is overcast, adding a moody ambiance to the scene. A close-up shot captures the details of the buses and the platform, emphasizing the dynamic energy of the terminal.
```

#### Example 2: single generation on the 14B model
This is the basic example for running inference on the 14B model with a single image. 
The output is saved to `outputs/video2world_14b.mp4` alongside the corresponding prompt at `outputs/video2world_14b.txt`.

```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict2/diffusion/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --input_image_or_video_path assets/video2world/input0.jpg \
    --num_input_frames 1 \
    --diffusion_transformer_dir Cosmos-Predict2-14B-Video2World \
    --offload_prompt_upsampler \
    --disable_prompt_upsampler \
    --disable_guardrail \
    --prompt "${PROMPT}" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --height 432 --width 768 --num_video_frames 81 \
    --num_steps 35 \
    --video_save_name video2world_14b
```

<!-- #### Example 2: single generation on the 14B model with model offloading
We run inference on the 14B model with offloading flags enabled. This is suitable for low-memory GPUs. Model offloading is also required for the 14B model to avoid OOM.
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict2/diffusion/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict2-14B-Video2World \
    --input_image_or_video_path assets/diffusion/video2world_input0.jpg \
    --num_input_frames 1 \
    --offload_tokenizer \
    --offload_diffusion_transformer \
    --offload_text_encoder_model \
    --offload_prompt_upsampler \
    --offload_guardrail_models \
    --video_save_name diffusion-video2world-14b
```

#### Example 3: single generation with multi-GPU inference
This example runs parallelized inference on a single prompt using 8 GPUs.
```bash
NUM_GPUS=8
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_predict2/diffusion/inference/video2world.py \
    --num_gpus ${NUM_GPUS} \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict2-2B-Video2World \
    --input_image_or_video_path assets/diffusion/video2world_input0.jpg \
    --num_input_frames 1 \
    --offload_prompt_upsampler \
    --video_save_name diffusion-video2world-2b
```

#### Example 4: batch generation
This example runs inference on a batch of prompts, provided through the `--batch_input_path` argument (path to a JSONL file).
Each line in the JSONL file must contain a `visual_input` field:
```json
{"visual_input": "path/to/video1.mp4"}
{"visual_input": "path/to/video2.mp4"}
```
Inference command (with 9 input frames):
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict2/diffusion/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict2-2B-Video2World \
    --batch_input_path assets/diffusion/batch_inputs/video2world_ps.jsonl \
    --num_input_frames 9 \
    --offload_prompt_upsampler \
    --video_save_folder diffusion-video2world-2b-batch
```

#### Example 5: batch generation without prompt upsampler
This example runs inference on a batch of prompts, provided through the `--batch_input_path` argument (path to a JSONL file).
The prompt upsampler is disabled, and thus each line in the JSONL file will need to include both `prompt` and `visual_input` fields.
```json
{"prompt": "prompt1", "visual_input": "path/to/video1.mp4"}
{"prompt": "prompt2", "visual_input": "path/to/video2.mp4"}
```
Inference command (with 9 input frames):
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict2/diffusion/inference/video2world.py \
    --checkpoint_dir checkpoints \
    --diffusion_transformer_dir Cosmos-Predict2-2B-Video2World \
    --batch_input_path assets/diffusion/batch_inputs/video2world_wo_ps.jsonl \
    --num_input_frames 9 \
    --disable_prompt_upsampler \
    --video_save_folder diffusion-video2world-2b-batch-wo-ps
``` -->


### Prompt engineering tips

To generate physically plausible videos with our video2world models, focus on realistic motion, consistent physics, and coherent environments. Start with a grounded scene. Describe the setting, characters, and actions clearly, such as "a person jogging on a sidewalk while cars pass by on the road." Ensure the initial frame is physically reasonable and sets up logical progression.

Avoid vague or surreal descriptions unless intentional. Instead, use real-world references and focus on how objects and characters interact naturally. Refine prompts based on output to correct issues like jerky motion or lighting inconsistencies. The more your prompt reflects real-world behavior and timing, the more believable the resulting video will be.
