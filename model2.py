# test model for medical image generation - https://huggingface.co/raman07/SD-finetuned-MIMIC-full

import os
from safetensors.torch import load_file
from diffusers.pipelines import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(sd_folder_path, revision="fp16")
exp_path = os.path.join('unet', 'diffusion_pytorch_model.safetensors')
state_dict = load_file(exp_path)

# Load the adapted U-Net
pipe.unet.load_state_dict(state_dict, strict=False)
pipe.to('cuda:0')

# Generate images with text prompts

TEXT_PROMPT = "Showing the subtrochanteric fracture in the porotic bone."
GUIDANCE_SCALE = 4
INFERENCE_STEPS = 75

result_image = pipe(
        prompt=TEXT_PROMPT,
        height=224,
        width=224,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=INFERENCE_STEPS,
    )

result_pil_image = result_image["images"][0]

result_pil_image.save("porotic_bone_fracture_model2.png")
