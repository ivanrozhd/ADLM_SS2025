#!pip install -r ../requirements.txt

from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
import torch.nn.functional as nnf
import numpy as np
import abc
from PIL import Image
from datetime import datetime
import os

import ptp_utils
import seq_aligner


class LocalBlend:
    def __call__(self, x_t, attention_store):
        """
        Input:  x_t: the latent representation of the image at timestep t
                attention_store: dictionary of attention maps collected during inference
        Logic:
                Extracts and reshapes attention maps from selected layers.
                Applies the alpha mask to emphasize selected words.
                Creates a binary spatial mask by pooling and normalizing.
                Applies the mask to blend only the changed regions of the image.
        """
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, self.MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(self, prompts: List[str], words: [List[List[str]]], MAX_NUM_WORDS, tokenizer, device, threshold=.3):
        """
        Input:  prompts: list of input text prompts
                words: list of target words (or word groups) to focus on
        """
        # alpha_layers: a binary mask tensor indicating which tokens (words) in each prompt should be emphasized
        self.MAX_NUM_WORDS = MAX_NUM_WORDS
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold

# A generic interface for manipulating attention
# The forward method is called in each attention layer of the diffusion model during the image generation
class AttentionControl(abc.ABC):

    # Called after each diffusion step — some controllers use this to track or adjust things like attention maps
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self): # handles low-resource mode by skipping attention edits if needed.
        #return self.num_att_layers if self.LOW_RESOURCE else 0
        return 0

    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str): # actual attention manipulation
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            #if self.LOW_RESOURCE:
            #    attn = self.forward(attn, is_cross, place_in_unet)
            #else:
            h = attn.shape[0]
            attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

# does nothing -> used when no manipulation is required
class EmptyControl(AttentionControl):
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn

# records attention maps during inference -> Useful for later inspection or editing
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        # saves attention maps by layer type (e.g., up_cross, down_self, etc.)
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        # aggregates maps across timesteps for averaging
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        # returns averaged maps over diffusion steps
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

# A generic interface (subclass of AttentionControl) to edit attention, especially cross-attention, during diffusion
class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        # applies the LocalBlend mask at each step (optional)
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(self, prompts, num_steps: int,
                cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]], # the fraction of steps to edit the cross attention maps. Can also be set to a dictionary [str:float] which specifies fractions for different words in the prompt
                self_replace_steps: Union[float, Tuple[float, float]], # the fraction of steps to replace the self attention maps
                tokenizer,device, # added by Ngoc
                local_blend: Optional[LocalBlend]
                ): # applies the LocalBlend mask at each step (optional)
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        # A temporal alpha tensor controlling the blending between base and modified attention
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

# Word Swap in the paper
class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        # replaces cross-attention using the mapper
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, 
                tokenizer,device, # added by Ngoc
                local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, tokenizer, device, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)

# Adding new Phrase in the paper
class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        # Blends old and new attention using alphas (blend coefficients)
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, tokenizer, device,
                local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, tokenizer, device, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])

# Attention Reweighting in the paper
class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        # strengthening or weakening the effect of specific words depending on the scalling tensor self.equalizer
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer, tokenizer, device, 
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, tokenizer, device, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller

# Creates a tensor (equalizer) that tells how much attention to apply to selected words
def get_equalizer(tokenizer, text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer

# Aggregates attention maps (either cross- or self-attention) over selected layers and steps
def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, prompts):
    """
    Input:  attention_store: Instance of AttentionStore holding averaged attention maps.
            res: Resolution of attention (e.g., 16 for 16×16 spatial attention).
            from_where: Layers to include (["up", "down", "mid"]).
            is_cross: If True, get cross-attention (text-to-image); else self-attention (image-to-image).
            select: Index of the prompt being visualized (supports batch prompts)
    """
    out = []
    attention_maps = attention_store.get_average_attention()
    #print("attention_maps:", attention_maps)
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()

# Displays the spatial attention map per token in a prompt (which parts of the image are associated with each single word).
def show_cross_attention(tokenizer, prompts, attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select, prompts)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    #ptp_utils.view_images(np.stack(images, axis=0))
    save_images_to_file(np.stack(images, axis=0), file_name_prefix="cross_attention_")

# Performs SVD on self-attention maps to reveal dominant components of spatial relationships -> useful for advanced inspection
def show_self_attention_comp(prompts, attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select, prompts).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    #ptp_utils.view_images(np.concatenate(images, axis=1))
    save_images_to_file(np.concatenate(images, axis=1), file_name_prefix="self_attention_")

# Runs Stable Diffusion with or without Prompt-to-Prompt control, displays output, and returns image + latent.
def run_and_display(prompts, controller, ldm_stable, NUM_DIFFUSION_STEPS, GUIDANCE_SCALE, latent=None, run_baseline=False, generator=None):
    """
    Input:  prompts: Text prompts (list of strings).
            controller: An instance of AttentionControl
            latent: Optional initial latent image.
            run_baseline: If True, runs default Stable Diffusion (without PtP) first.
            generator: Random seed control
    """
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = ptp_utils.text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator)
    #ptp_utils.view_images(images)
    save_images_to_file(images, file_name_prefix="run_")
    return images, x_t


def save_images_to_file(images, file_name_prefix, num_rows=1, offset_ratio=0.02, output_dir="generated_images"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if isinstance(images, list):
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    # Pad with white images if needed
    empty_image = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [img.astype(np.uint8) for img in images] + [empty_image] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255

    for i in range(num_rows):
        for j in range(num_cols):
            idx = i * num_cols + j
            image_[i * (h + offset): i * (h + offset) + h,
                   j * (w + offset): j * (w + offset) + w] = images[idx]

    # Convert to PIL and save with timestamp
    pil_img = Image.fromarray(image_)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{file_name_prefix}_{timestamp}.png"
    file_path = os.path.join(output_dir, filename)
    pil_img.save(file_path)

    print(f"[Saved]: {file_path}")