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
    "chest x-ray revealing  pneumonia",

]

fixed_seed = random.randint(0, 10000)
generator = torch.Generator(device=device).manual_seed(fixed_seed)

#fixed_seed = random.randint(0, 10000)
#generator = torch.Generator(device=device).manual_seed(fixed_seed)
image1 = pipe(prompts[0], generator=generator).images[0]
image1.save("img2_xray_pneumonia.png")
#image2 = pipe(prompts[1], generator=generator).images[0]
#image2.save("img2_xray_mass_right.png")






