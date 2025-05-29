from diffusers import StableDiffusionPipeline
import torch
import cross_attention_editting
import random

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)
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
#prompts = ["Chest X-ray demonstrating a pleural effusion"]
displayNumber = 0
fixed_seed = random.randint(0, 10000)
g_cpu = torch.Generator().manual_seed(fixed_seed)
#g_cpu = torch.Generator().manual_seed(8888) #specify a seed for the initial image


prompts = ["Chest computed tomography reveals lung nodules"]
displayNumber = 0
g_cpu = torch.Generator().manual_seed(1234) #specify a seed for the initial image
controller = cross_attention_editting.AttentionStore()
image, x_t = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable,
                                                NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE,
                                                latent=None, run_baseline=False, generator=g_cpu)
displayNumber +=1
cross_attention_editting.show_cross_attention(tokenizer, prompts, displayNumber, controller, res=16, from_where=("up", "down"), select=0) # select define which prompt




""" AttentionRefine: swap words """

#prompts = ["Computed tomography of the chest with a small tumor",
#            #"A painting of a lion eating a burger",
#            #"A painting of a cat eating a burger",
#            "Computed tomography of the chest with a large tumor"]
#lb = cross_attention_editting.LocalBlend(prompts, ("small", "large"), MAX_NUM_WORDS=MAX_NUM_WORDS, tokenizer=tokenizer, device=device)
#controller = cross_attention_editting.AttentionReplace(prompts, tokenizer=tokenizer, device=device, num_steps=NUM_DIFFUSION_STEPS,
#                                                    cross_replace_steps={"default_": 1., "large": .4}, #"lion": .4, "cat": .5,
#                                                    self_replace_steps=0.4, local_blend=lb)
#displayNumber += 1
#_ = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable,
#                                            NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE,
#                                            latent=None, run_baseline=False, generator=g_cpu)

prompts = ["chest x-ray of 26-year-old",  "chest x-ray of 26-year-old indicating infection"]


controller = cross_attention_editting.AttentionRefine(prompts, num_steps=NUM_DIFFUSION_STEPS, tokenizer=tokenizer, device=device,
                                                  cross_replace_steps=.8, self_replace_steps=.4)
displayNumber += 1
_ = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable,
                                          NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE,
                                         latent=x_t, run_baseline=True)

prompts = ["chest x-ray of 26-year-old",  "chest x-ray of 26-year-old where right lung is bigger than left lung"]

controller = cross_attention_editting.AttentionRefine(prompts, num_steps=NUM_DIFFUSION_STEPS, tokenizer=tokenizer, device=device,
                                                  cross_replace_steps=.8, self_replace_steps=.4)
displayNumber += 1
_ = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable,
                                          NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE,
                                         latent=x_t, run_baseline=True)


prompts = ["chest x-ray of 26-year-old",  "chest x-ray of 26-year-old demonstrating an abnormal heart"]

controller = cross_attention_editting.AttentionRefine(prompts, num_steps=NUM_DIFFUSION_STEPS, tokenizer=tokenizer, device=device,
                                                  cross_replace_steps=.8, self_replace_steps=.4)
displayNumber += 1
_ = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable,
                                          NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE,
                                         latent=x_t, run_baseline=True)


prompts = ["chest x-ray of 26-year-old",  "chest x-ray of 26-year-old showing a mass in the upper lobe"]

controller = cross_attention_editting.AttentionRefine(prompts, num_steps=NUM_DIFFUSION_STEPS, tokenizer=tokenizer, device=device,
                                                  cross_replace_steps=.8, self_replace_steps=.4)
displayNumber += 1
_ = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable,
                                          NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE,
                                         latent=x_t, run_baseline=True)