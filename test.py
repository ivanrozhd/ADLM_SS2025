import torch
from diffusers import StableDiffusionPipeline
import random
import os

# Load model
model_id = "Nihirc/Prompt2MedImage"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

# Brain CT prompts with minimal variation
prompts = ["Computed tomography of the chest  of 6-year-old ",
            "Computed tomography  of 26-year-old "
           ]

# Set a fixed seed for reproducibility
fixed_seed = random.randint(0, 10000)
generator = torch.Generator(device=device).manual_seed(fixed_seed)

# Output directory
output_dir = "images/safe_generated_images_ct_chest"
os.makedirs(output_dir, exist_ok=True)

# Generate and save images
for idx, prompt in enumerate(prompts):
    image = pipe(prompt, generator=generator).images[0]

    # Create a safe filename from the prompt
    filename = f"img{idx + 1}_{prompt.lower().replace(' ', '_').replace('.', '')}.png"

    # Save the image
    image.save(os.path.join(output_dir, filename))





