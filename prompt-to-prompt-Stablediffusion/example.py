from diffusers import StableDiffusionPipeline
import torch
import cross_attention_editting

MAX_NUM_WORDS = 77 # max number of tokens for the text input
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

#Loads the pre-trained Stable Diffusion model
ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
tokenizer = ldm_stable.tokenizer

# TODO: look into self attention show 

""" STORE AND VISUALIZE CROSS ATTENTION MAPS 
AttentionStore controller stores the (cross and self) attention maps of all prompts
Return:
    x_t : torch.Size([1, 4, 64, 64]) is a single noise seed sampled from a standard normal distribution -> independent of the prompts
    image : (amount of prompts, 512, 512, 3) 
"""
prompts = ["a photo of a house on a mountain", "A painting of a squirrel eating a burger"]
displayNumber = 0 
g_cpu = torch.Generator().manual_seed(8888) #specify a seed for the initial image 
controller = cross_attention_editting.AttentionStore()
image, x_t = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                                NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                                latent=None, run_baseline=False, generator=g_cpu) 
displayNumber +=1
cross_attention_editting.show_cross_attention(tokenizer, prompts, displayNumber, controller, res=16, from_where=("up", "down"), select=0) # select define which prompt 
displayNumber +=1
cross_attention_editting.show_cross_attention(tokenizer, prompts, displayNumber, controller, res=16, from_where=("up", "down"), select=1)

""" EDITTING TECHNIQUES 
Important arguments: 
    cross_replace_steps : the fraction of steps to edit the cross attention maps
        cross_replace_steps=.8 , or cross_replace_steps={"default_": .8} -> apply for all words
        cross_replace_steps={"default_": 1., "lion": .4, "cat": .5,} -> apply for specific words in edited prompts e.g. inject the "lion" token's attention during the early 40% of the diffusion process, then revert to the original

    local_blend (optional) : emphasize words representing thing that should be alter. LocalBlend only works for 2 prompts
"""

"""  AttentionReplace: swap words
Prompts have to be the same length, Can have multiple prompts """ 

prompts = ["A painting of a squirrel eating a burger",
            #"A painting of a lion eating a burger",
            #"A painting of a cat eating a burger",
            "A painting of a dog eating a burger"] 
lb = cross_attention_editting.LocalBlend(prompts, ("squirrel", "dog"), MAX_NUM_WORDS=MAX_NUM_WORDS, tokenizer=tokenizer, device=device) 
controller = cross_attention_editting.AttentionReplace(prompts, tokenizer=tokenizer, device=device, num_steps=NUM_DIFFUSION_STEPS, 
                                                    cross_replace_steps={"default_": 1., "dog": .4}, #"lion": .4, "cat": .5,
                                                    self_replace_steps=0.4, local_blend=lb)
displayNumber += 1
_ = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                            NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                            latent=x_t, run_baseline=False)

""" AttentionRefine: add words """
prompts = ["soup",
           "pea soup"]

lb = cross_attention_editting.LocalBlend(prompts, ("soup", "soup"), MAX_NUM_WORDS=MAX_NUM_WORDS, tokenizer=tokenizer, device=device) 
controller = cross_attention_editting.AttentionRefine(prompts, num_steps=NUM_DIFFUSION_STEPS, tokenizer=tokenizer, device=device,
                                                    cross_replace_steps=.8, self_replace_steps=.4
                                                    local_blend=lb)
displayNumber += 1
_ = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                            NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                            latent=x_t, run_baseline=True)

""" AttentionReweight: strengthen or weakens certain words 
* equalizer : define the words and the degree of strengthening or weakening
* can combine with other edit methods
"""  

prompts = ["pink bear riding a bicycle"] * 2

equalizer = cross_attention_editting.get_equalizer(tokenizer, prompts[1], ("pink",), (-1,))
lb = cross_attention_editting.LocalBlend(prompts, ("bicycle", "bicycle"), MAX_NUM_WORDS=MAX_NUM_WORDS, tokenizer=tokenizer, device=device) # apply the edit on the bikes 
controller = cross_attention_editting.AttentionReweight(prompts, num_steps=NUM_DIFFUSION_STEPS, tokenizer=tokenizer, device=device,
                                                    cross_replace_steps=.8, self_replace_steps=.4,
                                                    equalizer=equalizer, local_blend=lb)
displayNumber += 1
_ = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                            NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                            latent=x_t, run_baseline=False)


prompts = ["soup",
           "pea soup with croutons"] 

lb = cross_attention_editting.LocalBlend(prompts, ("soup", "soup"), MAX_NUM_WORDS=MAX_NUM_WORDS, tokenizer=tokenizer, device=device) 
controller_a = cross_attention_editting.AttentionRefine(prompts, num_steps=NUM_DIFFUSION_STEPS, tokenizer=tokenizer, device=device,
                                                    cross_replace_steps=.8, self_replace_steps=.4
                                                    local_blend=lb)

equalizer = cross_attention_editting.get_equalizer(tokenizer, prompts[1], ("croutons",), (3,))
controller = cross_attention_editting.AttentionReweight(prompts, num_steps=NUM_DIFFUSION_STEPS, tokenizer=tokenizer, device=device,
                               cross_replace_steps=.8, self_replace_steps=.4,
                               equalizer=equalizer, local_blend=lb,
                               controller=controller_a)
displayNumber += 1                               
_ = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                            NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                            latent=x_t, run_baseline=False)