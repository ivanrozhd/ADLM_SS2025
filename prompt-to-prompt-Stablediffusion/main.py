from diffusers import StableDiffusionPipeline
import torch
import cross_attention_editting
import random

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = StableDiffusionPipeline.from_pretrained("Nihirc/Prompt2MedImage").to(device) #torch_dtype=torch.float16)
tokenizer = ldm_stable.tokenizer
print(f"Using device: {device}")

# Hyper parameters
MAX_NUM_WORDS = 77 # max number of tokens for the text input
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
displayNumber = 0 




""" Mediastinal Mass 

random_seeds = [random.randint(1111, 9999) for _ in range(5)] + [5079,1320,3865,2313]
#random_seeds = [5079,1320,3865]
print(len(random_seeds), " random seeds for original images:", random_seeds)
for i, seed in enumerate(random_seeds):

        prompts = ["Chest CT scan", 
                # "Chest CT scan showing a mediastinal mass", 
                # "Chest CT scan showing a mass in the mediastinum", 
                "Chest CT scan showing a mass in the left mediastinum",
                "Chest CT scan showing a mass in the right mediastinum",
        ]

        g_cpu = torch.Generator().manual_seed(seed) #seed i
        controller = cross_attention_editting.AttentionStore()
        displayNumber += 1
        image, x_t = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                                        NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                                        latent=None, run_baseline=False, generator=g_cpu)
        #for j in range(len(prompts)):
        #    displayNumber += 1
        #    cross_attention_editting.show_cross_attention(tokenizer, prompts, displayNumber, controller, res=16, from_where=("up", "down"), select=j)
"""    

""" Abdomen """
prompts = ["CT scann of abdomen", 
           "CT scan of abdomen showing an anterior abdominal wall bulky abscess",
           "CT scan of abdomen showing a liver lesion"
           ]

# create five random seeds to generate three different images
#random_seeds = [random.randint(1111, 9999) for _ in range(20)] + [1888, 2025]
random_seeds = [6666]
print(len(random_seeds), " random seeds for original images:", random_seeds)

for i, seed in enumerate(random_seeds):
    g_cpu = torch.Generator().manual_seed(seed) #seed i
    controller = cross_attention_editting.AttentionStore()
    displayNumber += 1
    image, x_t = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                                NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                                latent=None, run_baseline=False, generator=g_cpu)

    for j in range(len(prompts)):
        displayNumber += 1
        cross_attention_editting.show_cross_attention_per_word(tokenizer, prompts, displayNumber, controller, res=16, from_where=("up", "down"), select=j)
 

""" Abdomen, Anterior Abdominal Wall Bulky Abscess 

prompts = ["CT scan of abdomen", 
        "CT scan of abdomen showing an anterior abdominal wall bulky abscess"]

#lb = cross_attention_editting.LocalBlend(prompts, ("soup", "soup"), MAX_NUM_WORDS=MAX_NUM_WORDS, tokenizer=tokenizer, device=device) 
controller = cross_attention_editting.AttentionRefine(prompts, num_steps=NUM_DIFFUSION_STEPS, tokenizer=tokenizer, device=device, 
                                                    cross_replace_steps=1., self_replace_steps=.4, 
                                                    local_blend=None)
displayNumber += 1
_ = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                            NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                            latent=x_t, run_baseline=False)

prompts = ["CT scan of abdomen showing an anterior abdominal wall bulky abscess of 12 cm x 14 cm", 
        "CT scan of abdomen showing an anterior abdominal wall bulky abscess of 1 cm x 2 cm"]
lb = cross_attention_editting.LocalBlend(prompts, ("abscess", "abscess"), MAX_NUM_WORDS=MAX_NUM_WORDS, tokenizer=tokenizer, device=device) # TODO: local blend auf liver tumor? 
controller = cross_attention_editting.AttentionReplace(prompts, tokenizer=tokenizer, device=device, num_steps=NUM_DIFFUSION_STEPS, 
                                                    cross_replace_steps={"default_": 1.},
                                                    self_replace_steps=0.4, local_blend=lb)
displayNumber += 1
_ = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                            NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                            latent=x_t, run_baseline=False)

for weight in [-30, -20, -15, -10, -5, 5, 10, 20, 30, 50]:   
    prompts = ["CT scan of abdomen showing an anterior abdominal wall bulky abscess"] * 2
    equalizer = cross_attention_editting.get_equalizer(tokenizer, prompts[1], ("abscess",), (weight,))
    #lb = cross_attention_editting.LocalBlend(prompts, ("abscess", "abscess"), MAX_NUM_WORDS=MAX_NUM_WORDS, tokenizer=tokenizer, device=device) 
    controller = cross_attention_editting.AttentionReweight(prompts, num_steps=NUM_DIFFUSION_STEPS, tokenizer=tokenizer, device=device,
                                                        cross_replace_steps=.8, self_replace_steps=.4,
                                                        equalizer=equalizer, local_blend=None)
    displayNumber += 1
    _ = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                                NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                                latent=x_t, run_baseline=False)
"""   

""" Abdomen, liver tumor  

prompts = ["CT scann of abdomen"]
g_cpu = torch.Generator().manual_seed(6666) #seed i
controller = cross_attention_editting.AttentionStore()
displayNumber += 1
image, x_t = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                            NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                            latent=None, run_baseline=False, generator=g_cpu)


prompts = ["A CT scan of abdomen", 
           "A CT scan of abdomen showing a liver tumor"]

#lb = cross_attention_editting.LocalBlend(prompts, ("soup", "soup"), MAX_NUM_WORDS=MAX_NUM_WORDS, tokenizer=tokenizer, device=device) 
controller = cross_attention_editting.AttentionRefine(prompts, num_steps=NUM_DIFFUSION_STEPS, tokenizer=tokenizer, device=device, 
                                                      cross_replace_steps=.8, self_replace_steps=.4, 
                                                      local_blend=None)
displayNumber += 1
_ = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                            NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                            latent=x_t, run_baseline=False)
    

prompts = ["A CT scan of abdomen showing a liver tumor of size 20 cm", 
           "A CT scan of abdomen showing a liver tumor of size 5 cm"]
lb = cross_attention_editting.LocalBlend(prompts, ("tumor", "tumor"), MAX_NUM_WORDS=MAX_NUM_WORDS, tokenizer=tokenizer, device=device) # TODO: local blend auf liver tumor? 
controller = cross_attention_editting.AttentionReplace(prompts, tokenizer=tokenizer, device=device, num_steps=NUM_DIFFUSION_STEPS, 
                                                    cross_replace_steps={"default_": 1.}, #"lion": .4, "cat": .5,
                                                    self_replace_steps=0.4, local_blend=lb)
displayNumber += 1
_ = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                            NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                            latent=x_t, run_baseline=False)


prompts = ["A CT scan of abdomen showing a large liver tumor", 
           "A CT scan of abdomen showing a small liver tumor", ]
lb = cross_attention_editting.LocalBlend(prompts, ("tumor", "tumor"), MAX_NUM_WORDS=MAX_NUM_WORDS, tokenizer=tokenizer, device=device) 
controller = cross_attention_editting.AttentionReplace(prompts, tokenizer=tokenizer, device=device, num_steps=NUM_DIFFUSION_STEPS, 
                                                    cross_replace_steps={"default_": 1.}, #"lion": .4, "cat": .5,
                                                    self_replace_steps=0.4, local_blend=lb)
displayNumber += 1
_ = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                            NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                            latent=x_t, run_baseline=False)

prompts = ["A CT scan of abdomen showing a liver tumor of size 20 cm", 
           "A CT scan of abdomen showing a liver tumor of size 5 cm"]
lb = cross_attention_editting.LocalBlend(prompts, ("liver tumor", "liver tumor"), MAX_NUM_WORDS=MAX_NUM_WORDS, tokenizer=tokenizer, device=device) # TODO: local blend auf liver tumor? 
controller = cross_attention_editting.AttentionReplace(prompts, tokenizer=tokenizer, device=device, num_steps=NUM_DIFFUSION_STEPS, 
                                                    cross_replace_steps={"default_": 1.}, #"lion": .4, "cat": .5,
                                                    self_replace_steps=0.4, local_blend=lb)
displayNumber += 1
_ = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                            NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                            latent=x_t, run_baseline=False)


prompts = ["A CT scan of abdomen showing a large liver tumor", 
           "A CT scan of abdomen showing a small liver tumor", ]
lb = cross_attention_editting.LocalBlend(prompts, ("liver tumor", "liver tumor"), MAX_NUM_WORDS=MAX_NUM_WORDS, tokenizer=tokenizer, device=device) 
controller = cross_attention_editting.AttentionReplace(prompts, tokenizer=tokenizer, device=device, num_steps=NUM_DIFFUSION_STEPS, 
                                                    cross_replace_steps={"default_": 1.}, #"lion": .4, "cat": .5,
                                                    self_replace_steps=0.4, local_blend=lb)
displayNumber += 1
_ = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                            NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                            latent=x_t, run_baseline=False)


for weight in [-3, -2.5, -2, -1.5, -1, 1, 5, 10, 15]:   
    prompts = ["A CT scan of abdomen showing a liver cyst"] * 2
    equalizer = cross_attention_editting.get_equalizer(tokenizer, prompts[1], ("cyst",), (weight,))
    lb = cross_attention_editting.LocalBlend(prompts, ("cyst", "cyst"), MAX_NUM_WORDS=MAX_NUM_WORDS, tokenizer=tokenizer, device=device) 
    controller = cross_attention_editting.AttentionReweight(prompts, num_steps=NUM_DIFFUSION_STEPS, tokenizer=tokenizer, device=device,
                                                        cross_replace_steps=.8, self_replace_steps=.4,
                                                        equalizer=equalizer, local_blend=lb)
    displayNumber += 1
    _ = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                                NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                                latent=x_t, run_baseline=False)
"""