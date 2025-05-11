import torch
from diffusers import StableDiffusionPipeline
import random

# Load model
model_id = "Nihirc/Prompt2MedImage"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

# Brain CT prompts with minimal variation
prompts = [
    "CT scan of the lungs showing a tumor nodule", # slightly different
]

fixed_seed = 1311 #random.randint(0, 10000)
generator = torch.Generator(device=device).manual_seed(fixed_seed)

#fixed_seed = random.randint(0, 10000)
#generator = torch.Generator(device=device).manual_seed(fixed_seed)
image2 = pipe(prompts[0], generator=generator).images[0]
image2.save("img2_fix_lungs.png")





