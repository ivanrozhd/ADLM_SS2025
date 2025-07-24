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


""" Lung Mass & Nodules """ 
#random_seeds = [random.randint(1111, 9999) for _ in range(1)]
random_seeds =  [1122, 1469, 2790, 3933, 4170, 4462, 5210, 5935, 7177]

print(len(random_seeds), " random seeds for original images:", random_seeds)
for i, seed in enumerate(random_seeds):

        prompts = ["Chest CT scan"] #["Chest CT scan showing lung nodules"] #["Chest CT scan showing multiple lung nodules"] 

        g_cpu = torch.Generator().manual_seed(seed) #seed i
        controller = cross_attention_editting.AttentionStore()
        displayNumber += 1
        image, x_t = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                                        NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                                        latent=None, run_baseline=False, generator=g_cpu)
        #for j in range(len(prompts)):
        #    displayNumber += 1
        #    cross_attention_editting.show_cross_attention_per_word(tokenizer, prompts, displayNumber, controller, res=16, from_where=("up", "down"), select=j)
        #cross_attention_editting.save_images_separately(prompts, image, image_name=f"{seed}_original_reweight_lung_nodules", use_case="reweight_lung_nodules")
        #cross_attention_editting.save_images_separately(prompts, image, image_name=f"{seed}_original_remove_lung_nodules", use_case="remove_lung_nodules")
        cross_attention_editting.save_images_separately(prompts, image, image_name=f"{seed}_original_add_lung_nodules", use_case="add_lung_nodules")
        

        ### ADDING OR REMOVING LUNG NODULES (Loop through different cross_replace_steps and self_replace_steps) ###
        for cross_replace_steps in [0.2, 0.5, 0.8]: 
                for self_replace_steps in [0.2, 0.4, 0.6]:
                        prompts = ["Chest CT scan", 
                                "Chest CT scan showing lung nodules"]
                        controller = cross_attention_editting.AttentionRefine(prompts, num_steps=NUM_DIFFUSION_STEPS, tokenizer=tokenizer, device=device, 
                                                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                                                        local_blend=None)
                        displayNumber += 1
                        images, _ = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                                                NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                                                latent=x_t, run_baseline=False)
                        
                        cross_attention_editting.save_images_separately(prompts, images, image_name=f"{seed}_{cross_replace_steps}_{self_replace_steps}_add_lung_nodules", use_case="add_lung_nodules")
                        
                        """
                        prompts = ["Chest CT scan showing lung nodules", 
                                "Chest CT scan"]
                        controller = cross_attention_editting.AttentionRefine(prompts, num_steps=NUM_DIFFUSION_STEPS, tokenizer=tokenizer, device=device, 
                                                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                                                        local_blend=None)
                        displayNumber += 1
                        images, _ = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                                                NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                                                latent=x_t, run_baseline=False)
                        cross_attention_editting.save_images_separately(prompts, images, image_name=f"{seed}_{cross_replace_steps}_{self_replace_steps}_remove_lung_nodules", use_case="remove_lung_nodules")
                        """
        
        ### REWEIGHTING LUNG NODULES (Loop through different reweighting weights) ###
        """
        for weight in [-15, -10, -5, 5, 10, 15]:   
                prompts = ["Chest CT scan showing multiple lung nodules"] * 2
                equalizer = cross_attention_editting.get_equalizer(tokenizer, prompts[1], ("nodules",), (weight,))
                #lb = cross_attention_editting.LocalBlend(prompts, ("lung", "lung"), MAX_NUM_WORDS=MAX_NUM_WORDS, tokenizer=tokenizer, device=device) 
                controller = cross_attention_editting.AttentionReweight(prompts, num_steps=NUM_DIFFUSION_STEPS, tokenizer=tokenizer, device=device,
                                                                        cross_replace_steps=.8, self_replace_steps=.4,
                                                                        equalizer=equalizer, local_blend=None)
                displayNumber += 1
                images, _ = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                                                NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                                                latent=x_t, run_baseline=False)   
                cross_attention_editting.save_images_separately(prompts, images, image_name=f"{seed}_{weight}_reweight_lung_nodules", use_case="reweight_lung_nodules")
        """


""" cardiomegaly: heart enlargement """ 
""" 
#random_seeds = [random.randint(1111, 9999) for _ in range(10)]
random_seeds = [1122, 1469, 2369, 2590, 2790, 3933, 4170, 5727, 6935, 7177, 7290]
print(len(random_seeds), " random seeds for original images:", random_seeds)
for i, seed in enumerate(random_seeds):

        prompts = [#"Chest CT scan",
                "Chest CT scan showing a cardiomegaly"
        ]

        g_cpu = torch.Generator().manual_seed(seed) #seed i
        controller = cross_attention_editting.AttentionStore()
        displayNumber += 1
        image, x_t = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                                        NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                                        latent=None, run_baseline=False, generator=g_cpu) #save_images_side_by_side=True
        #for j in range(len(prompts)):
        #    displayNumber += 1
        #    cross_attention_editting.show_cross_attention_per_word(tokenizer, prompts, displayNumber, controller, res=16, from_where=("up", "down"), select=j)
        
        #cross_attention_editting.save_images_separately(prompts, image, image_name=f"{seed}_original_add_cardiomegaly", use_case="add_cardiomegaly")
        #cross_attention_editting.save_images_separately(prompts, image, image_name=f"{seed}_original_remove_cardiomegaly", use_case="remove_cardiomegaly")
        #cross_attention_editting.save_images_separately(prompts, image, image_name=f"{seed}_original_reweight_cardiomegaly", use_case="reweight_cardiomegaly")

        
        ### ADDING OR REMOVING CARDIOMEGALY (Loop through different cross_replace_steps and self_replace_steps) ###
        for cross_replace_steps in [0.2, 0.5, 0.8]:
                for self_replace_steps in [0.2, 0.4, 0.6]:
                        
                        prompts = ["Chest CT scan", 
                                "Chest CT scan showing a cardiomegaly"]
                        controller = cross_attention_editting.AttentionRefine(prompts, num_steps=NUM_DIFFUSION_STEPS, tokenizer=tokenizer, device=device, 
                                                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                                                        local_blend=None)
                        displayNumber += 1
                        images, _ = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                                                NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                                                latent=x_t, run_baseline=False)
                        
                        cross_attention_editting.save_images_separately(prompts, images, image_name=f"{seed}_{cross_replace_steps}_{self_replace_steps}_add_cardiomegaly", use_case="add_cardiomegaly")
                        
                        prompts = ["Chest CT scan showing a cardiomegaly", 
                                "Chest CT scan"]
                        controller = cross_attention_editting.AttentionRefine(prompts, num_steps=NUM_DIFFUSION_STEPS, tokenizer=tokenizer, device=device, 
                                                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                                                        local_blend=None)
                        displayNumber += 1
                        images, _ = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                                                NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                                                latent=x_t, run_baseline=False)
                        cross_attention_editting.save_images_separately(prompts, images, image_name=f"{seed}_{cross_replace_steps}_{self_replace_steps}_remove_cardiomegaly", use_case="remove_cardiomegaly")
                        

        ### REWEIGHTING CARDIOMEGALY (Loop through different reweighting weights) ###
        for weight in [-1, -2, -3, 1, 2, 3]:   
                prompts = ["Chest CT scan showing a cardiomegaly"] * 2
                equalizer = cross_attention_editting.get_equalizer(tokenizer, prompts[1], ("cardiomegaly",), (weight,))
                #lb = cross_attention_editting.LocalBlend(prompts, ("lung", "lung"), MAX_NUM_WORDS=MAX_NUM_WORDS, tokenizer=tokenizer, device=device) 
                controller = cross_attention_editting.AttentionReweight(prompts, num_steps=NUM_DIFFUSION_STEPS, tokenizer=tokenizer, device=device,
                                                                        cross_replace_steps=.8, self_replace_steps=.4,
                                                                        equalizer=equalizer, local_blend=None)
                displayNumber += 1
                images, _ = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                                                NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                                                latent=x_t, run_baseline=False)   
                cross_attention_editting.save_images_separately(prompts, images, image_name=f"{seed}_{weight}_reweight_cardiomegaly", use_case="reweight_cardiomegaly")
"""

