# test model for medical image generation - https://huggingface.co/raman07/SD-finetuned-MIMIC-full

import os
from safetensors.torch import load_file
from diffusers.pipelines import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("raman07/SD-finetuned-MIMIC-full")
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
