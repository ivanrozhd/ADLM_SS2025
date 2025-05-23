from diffusers import StableDiffusionPipeline
import torch
import cross_attention_editting


MAX_NUM_WORDS = 77 # max number of tokens for the text input
#LOW_RESOURCE = False # If True, the code may use optimizations to reduce memory or computation cost, useful for running on limited hardware.
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

#Loads the pre-trained Stable Diffusion model
ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device) #torch_dtype=torch.float16)
#ldm_stable = StableDiffusionPipeline.from_pretrained("Nihirc/Prompt2MedImage").to(device) #torch_dtype=torch.float16)
tokenizer = ldm_stable.tokenizer


################# TEST PRINT OUT CROSS ATTENTION LAYERS #######################
g_cpu = torch.Generator().manual_seed(8888)
prompts = ["A painting of a squirrel eating a burger"]
controller = cross_attention_editting.AttentionStore()
image, x_t = cross_attention_editting.run_and_display(prompts, controller, ldm_stable, NUM_DIFFUSION_STEPS, GUIDANCE_SCALE, latent=None, run_baseline=False, generator=g_cpu)
cross_attention_editting.show_cross_attention(tokenizer, controller, res=16, from_where=("up", "down"))


################## TEST AttentionReplace ######################
prompts = ["A painting of a squirrel eating a burger",
           "A painting of a lion eating a burger"]

controller = cross_attention_editting.AttentionReplace(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, self_replace_steps=0.4, tokenizer=tokenizer, device=device)
_ = cross_attention_editting.run_and_display(prompts, controller, ldm_stable, NUM_DIFFUSION_STEPS, GUIDANCE_SCALE, latent=x_t, run_baseline=True, generator=g_cpu)


################## TEST AttentionReplace ######################
prompts = ["a photo of a house on a mountain",
           "a photo of a house on a mountain at fall"]

controller = cross_attention_editting.AttentionRefine(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, tokenizer=tokenizer, device=device, self_replace_steps=.4)
_ = cross_attention_editting.run_and_display(prompts, controller,  ldm_stable, NUM_DIFFUSION_STEPS, GUIDANCE_SCALE, latent=x_t, run_baseline=True, generator=g_cpu)


################## TEST LocalBlend and AttentionReweight ######################
prompts = ["soup",
           "pea soup with croutons"] 
lb = cross_attention_editting.LocalBlend(prompts, ("soup", "soup"), MAX_NUM_WORDS=MAX_NUM_WORDS, tokenizer=tokenizer, device=device)
controller = cross_attention_editting.AttentionRefine(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, tokenizer=tokenizer, device=device,
                               self_replace_steps=.4, local_blend=lb)
_ = cross_attention_editting.run_and_display(prompts, controller, ldm_stable, NUM_DIFFUSION_STEPS, GUIDANCE_SCALE, latent=x_t, run_baseline=False)


################## TEST Combine multiple Editings ######################
prompts = ["soup",
           "pea soup with croutons"]

lb = cross_attention_editting.LocalBlend(prompts, ("soup", "soup"), MAX_NUM_WORDS=MAX_NUM_WORDS, tokenizer=tokenizer, device=device)

controller_a = cross_attention_editting.AttentionRefine(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, tokenizer=tokenizer, device=device,
                               self_replace_steps=.4, local_blend=lb)

### pay 3 times more attention to the word "croutons"
equalizer = cross_attention_editting.get_equalizer(tokenizer, prompts[1], ("croutons",), (3,))
controller = cross_attention_editting.AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, tokenizer=tokenizer, device=device,
                               self_replace_steps=.4, equalizer=equalizer, local_blend=lb,
                               controller=controller_a)
_ = cross_attention_editting.run_and_display(prompts, controller, ldm_stable, NUM_DIFFUSION_STEPS, GUIDANCE_SCALE, latent=x_t, run_baseline=False)