import torch
from diffusers import StableDiffusionPipeline

model_id = "Nihirc/Prompt2MedImage"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

prompt = "Showing the subtrochanteric fracture in the porotic bone."
image = pipe(prompt).images[0]  
    
image.save("porotic_bone_fracture.png")
