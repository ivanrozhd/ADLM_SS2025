from diffusers import StableDiffusionPipeline
import torch
import cross_attention_editting
import random
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = StableDiffusionPipeline.from_pretrained("Nihirc/Prompt2MedImage").to(device) #torch_dtype=torch.float16)
tokenizer = ldm_stable.tokenizer

# Hyper parameters
MAX_NUM_WORDS = 77 # max number of tokens for the text input
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5


# TODO: look into self attention show

""" STORE AND VISUALIZE CROSS ATTENTION MAPS 
AttentionStore controller stores the (cross and self) attention maps of all prompts
Return:
    x_t : torch.Size([1, 4, 64, 64]) is a single noise seed sampled from a standard normal distribution -> independent of the prompts
    image : (amount of prompts, 512, 512, 3) 
"""
prompts = ["chest x-ray of 6-year-old child", "chest x-ray of 26-year-old", "chest x-ray of 1-year-old child"]
displayNumber = 0
fixed_seed = random.randint(0, 10000)
g_cpu = torch.Generator().manual_seed(fixed_seed) #specify a seed for the initial image
controller = cross_attention_editting.AttentionStore()
image, x_t = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable,
                                                NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE,
                                                latent=None, run_baseline=False, generator=g_cpu)
displayNumber +=1
cross_attention_editting.show_cross_attention(tokenizer, prompts, displayNumber, controller, res=16, from_where=("up", "down"), select=0) # select define which prompt
displayNumber +=1
cross_attention_editting.show_cross_attention(tokenizer, prompts, displayNumber, controller, res=16, from_where=("up", "down"), select=1)
displayNumber +=1
cross_attention_editting.show_cross_attention(tokenizer, prompts, displayNumber, controller, res=16, from_where=("up", "down"), select=2)
