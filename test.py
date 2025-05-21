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
    "High resolution CT chest slices of mild bronchiectasis of the right lower lobe with partial collapse of right middle lobe",
    "High resolution CT chest slices of mild bronchiectasis of the right lower lobe with total collapse of right middle lobe",
]

fixed_seed = random.randint(0, 10000)
generator = torch.Generator(device=device).manual_seed(fixed_seed)

#fixed_seed = random.randint(0, 10000)
#generator = torch.Generator(device=device).manual_seed(fixed_seed)
#image1 = pipe(prompts[0], generator=generator).images[0]
#image1.save("img1_fix_lungs_healthy.png")
image2 = pipe(prompts[0], generator=generator).images[0]
image2.save("img2_fix_lungs_partially.png")
image3 = pipe(prompts[1], generator=generator).images[0]
image3.save("img3_fix_lungs_fully.png")





