from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
#import cross_attention_editting
import random
import null_text_inversion
import os
import ptp_utils
import cross_attention_editting

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

ldm_stable = StableDiffusionPipeline.from_pretrained("Nihirc/Prompt2MedImage", scheduler=scheduler).to(device) #torch_dtype=torch.float16)
tokenizer = ldm_stable.tokenizer

print(f"Using device: {device}")

# Hyper parameters
MAX_NUM_WORDS = 77 # max number of tokens for the text input
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
displayNumber = 200 
#MY_TOKEN = ''


null_inversion = null_text_inversion.NullInversion(ldm_stable, device=device, NUM_DDIM_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE)


image_path = "PMC3563702_trd-74-37-g001.jpg"
#image_path = "gnochi_mirror.jpeg"
prompt = "Chest computed tomography scan showed a mass lesion on the upper lobe of the right lung"
#prompt = "a cat sitting next to a mirror"

(image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0))
#print("Modify or remove offsets according to your image!")

prompts = [prompt, 
           "Chest computed tomography scan showed a healthy lung", 
           "Chest computed tomography scan showed a mass lesion on the upper lobe of the right lung and a mass in the left lung", 
           "Chest computed tomography scan showed a small mass lesion on the upper lobe of the right lung",
           "Chest computed tomography scan showed a large mass lesion on the upper lobe of the right lung"]

controller = cross_attention_editting.AttentionStore()
displayNumber += 1
image, x_t = cross_attention_editting.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                                NUM_DIFFUSION_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                                latent=x_t, run_baseline=False)
for j in range(len(prompts)):
    displayNumber += 1
    cross_attention_editting.show_cross_attention_per_word(tokenizer, prompts, displayNumber, controller, res=16, from_where=("up", "down"), select=j)



"""
controller = null_text_inversion.AttentionStore()
image_inv, x_t = null_text_inversion.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                                     NUM_DDIM_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                                     latent=x_t, uncond_embeddings=uncond_embeddings)

#ptp_utils.view_images([image_gt, image_enc, image_inv[0]])
import os
import numpy as np
from PIL import Image

def save_images(images, num_rows=1, offset_ratio=0.02, save_path="./test_inversion_output/output.png"):
    if isinstance(images, list):
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    # Create white images to fill in empty slots
    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows

    # Create a white canvas for the final output
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255

    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h,
                   j * (w + offset): j * (w + offset) + w] = images[i * num_cols + j]

    pil_img = Image.fromarray(image_)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the image
    pil_img.save(save_path)

# save the three images to folder "test_inversion_output"
print("showing from left to right: the ground truth image, the vq-autoencoder reconstruction, the null-text inverted image")
save_images([image_gt, image_enc, image_inv[0]], save_path="./test_inversion_output/output_0.png")
save_images([image_gt, image_enc, image_inv[1]], save_path="./test_inversion_output/output_1.png")
save_images([image_gt, image_enc, image_inv[2]], save_path="./test_inversion_output/output_2.png")
save_images([image_gt, image_enc, image_inv[3]], save_path="./test_inversion_output/output_3.png")
save_images([image_gt, image_enc, image_inv[4]], save_path="./test_inversion_output/output_4.png")

print("show cross attention per word for the inverted image")
null_text_inversion.show_cross_attention(prompts, tokenizer, controller, 16, ["up", "down"])
"""

""" Cat Example 
prompts = ["a cat sitting next to a mirror",
           "a silver cat sculpture sitting next to a mirror"]

cross_replace_steps = {'default_': .8, }
self_replace_steps = .6
blend_word = ((('cat',), ("cat",))) # for local edit
eq_params = {"words": ("silver", 'sculpture', ), "values": (2,2,)}  # amplify attention to the words "silver" and "sculpture" by *2 
 
controller = null_text_inversion.make_controller(prompts, MAX_NUM_WORDS, NUM_DIFFUSION_STEPS, device, tokenizer, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
displayNumber += 1
images, _ = null_text_inversion.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                                NUM_DDIM_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                                latent=x_t, uncond_embeddings=uncond_embeddings)

prompts = ["a cat sitting next to a mirror", 
            "a mirror"]

cross_replace_steps = {'default_': .8, }
self_replace_steps = .6
blend_word = None #((('cat',), ("cat",))) # for local edit
eq_params = None #{"words": ['cat',], "values": [-10,]}  
 
controller = null_text_inversion.make_controller(prompts, MAX_NUM_WORDS, NUM_DIFFUSION_STEPS, device, tokenizer, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
displayNumber += 1
images, _ = null_text_inversion.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                                NUM_DDIM_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                                latent=x_t, uncond_embeddings=uncond_embeddings)
"""

"""
# make cross_replace_steps as .8 or .3 and self_replace_steps as .6 or .2, make the loop through the 4 combinations
for cross_replace_steps in [{'default_': .8}, {'default_': .3}]:
    for self_replace_steps in [.6, .2]:

        # TODO: add "a mass in the left lung" 
        prompts = ["Chest computed tomography scan showed a mass lesion on the upper lobe of the right lung",
                   "Chest computed tomography scan showed a mass lesion on the upper lobe of the right lung and a mass in the left lung"]

        blend_word = None
        eq_params = None
        controller = null_text_inversion.make_controller(prompts, MAX_NUM_WORDS, NUM_DIFFUSION_STEPS, device, tokenizer, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
        displayNumber += 1
        images, _ = null_text_inversion.run_and_display(prompts, displayNumber, controller, ldm_stable, 
                                                        NUM_DDIM_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                                        latent=x_t, uncond_embeddings=uncond_embeddings)
        
        # TODO: add "small" to the prompts
        prompts = ["Chest computed tomography scan showed a mass lesion on the upper lobe of the right lung",
                   "Chest computed tomography scan showed a small mass lesion on the upper lobe of the right lung"]
        
        blend_word = ((('mass lesion',), ("mass lesion",))) 
        eq_params = {"words": ("small", ), "values": (5,)}  
        controller = null_text_inversion.make_controller(prompts, MAX_NUM_WORDS, NUM_DIFFUSION_STEPS, device, tokenizer, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
        displayNumber += 1
        images, _ = null_text_inversion.run_and_display(prompts, displayNumber, controller
                                                        , ldm_stable, 
                                                        NUM_DDIM_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                                        latent=x_t, uncond_embeddings=uncond_embeddings)

        # TODO: add "large" to the prompts
        prompts = ["Chest computed tomography scan showed a mass lesion on the upper lobe of the right lung",
                   "Chest computed tomography scan showed a large mass lesion on the upper lobe of the right lung"]

        blend_word = ((('mass lesion',), ("mass lesion",))) 
        eq_params = {"words": ("large", ), "values": (5,)}  
        controller = null_text_inversion.make_controller(prompts, MAX_NUM_WORDS, NUM_DIFFUSION_STEPS, device, tokenizer, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
        displayNumber += 1
        images, _ = null_text_inversion.run_and_display(prompts, displayNumber, controller
                                                        , ldm_stable, 
                                                        NUM_DDIM_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                                        latent=x_t, uncond_embeddings=uncond_embeddings)

    
        # TODO: change to "Chest computed tomography scan showed a healthy lung"
        prompts = ["Chest computed tomography scan showed a mass lesion on the upper lobe of the right lung",
                   "Chest computed tomography scan showed a healthy lung"]
        blend_word = None
        eq_params = {"words": ("healthy", ), "values": (5,)}  
        controller = null_text_inversion.make_controller(prompts, MAX_NUM_WORDS, NUM_DIFFUSION_STEPS, device, tokenizer, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
        displayNumber += 1
        images, _ = null_text_inversion.run_and_display(prompts, displayNumber, controller
                                                        , ldm_stable, 
                                                        NUM_DDIM_STEPS=NUM_DIFFUSION_STEPS, GUIDANCE_SCALE=GUIDANCE_SCALE, 
                                                        latent=x_t, uncond_embeddings=uncond_embeddings)
"""