from diffusers import StableDiffusionPipeline
import torch
import cross_attention_editting


MAX_NUM_WORDS = 77 # max number of tokens for the text input
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

#Loads the pre-trained Stable Diffusion model
ldm_stable = StableDiffusionPipeline.from_pretrained("Nihirc/Prompt2MedImage").to(device) #torch_dtype=torch.float16)
tokenizer = ldm_stable.tokenizer


################# TEST PRINT OUT CROSS ATTENTION LAYERS #######################
g_cpu = torch.Generator().manual_seed(4444)
prompts = ["MRI left brain tumor"]
controller = cross_attention_editting.AttentionStore()
image, x_t = cross_attention_editting.run_and_display(prompts, controller, ldm_stable, 
                                                NUM_DIFFUSION_STEPS, GUIDANCE_SCALE, latent=None, run_baseline=False, 
                                                generator=g_cpu) #cross_replace_steps={"default_": 1., "lion": .4}, self_replace_steps=0.4, local_blend=lb
cross_attention_editting.show_cross_attention(tokenizer, prompts, controller, res=16, from_where=("up", "down"))


################## TEST AttentionReplace ######################
prompts = ["MRI left brain tumor",
           "MRI right brain tumor"]

controller = cross_attention_editting.AttentionReplace(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, self_replace_steps=0.4, tokenizer=tokenizer, device=device)
_ = cross_attention_editting.run_and_display(prompts, controller, ldm_stable, NUM_DIFFUSION_STEPS, GUIDANCE_SCALE, latent=x_t, run_baseline=True, generator=g_cpu)


################## TEST AttentionReplace ######################
prompts = ["MRI left brain",
           "MRI left brain tumor"]

controller = cross_attention_editting.AttentionRefine(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, tokenizer=tokenizer, device=device, self_replace_steps=.4)
_ = cross_attention_editting.run_and_display(prompts, controller,  ldm_stable, NUM_DIFFUSION_STEPS, GUIDANCE_SCALE, latent=x_t, run_baseline=True, generator=g_cpu)


################## TEST AttentionReweight ######################
prompts = ["MRI left brain tumor"] * 2

equalizer = cross_attention_editting.get_equalizer(tokenizer, prompts[1], ("tumor",), (-5,))
controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, tokenizer=tokenizer, device=device,
                               self_replace_steps=.4,
                               equalizer=equalizer)

_ = cross_attention_editting.run_and_display(prompts, controller, ldm_stable, NUM_DIFFUSION_STEPS, GUIDANCE_SCALE, latent=x_t, run_baseline=False)
