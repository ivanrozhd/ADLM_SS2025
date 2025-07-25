import os 
from PIL import Image, ImageOps
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
from scipy.ndimage import map_coordinates, gaussian_filter
import random


#------------------------------Augmentation Techniques---------------------------------------------------
def elastic_deformation(image, alpha=15, sigma=3):
    random_state = np.random.RandomState(None)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant") * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant") * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy).reshape(-1), (x + dx).reshape(-1)
    return map_coordinates(image, indices, order=1).reshape(shape)


def direct_brightness_contrast(slice_img, brightness_range=(0.9, 1.1), contrast_range=(0.9, 1.2)):

    brightness_factor = random.uniform(*brightness_range)
    contrast_factor = random.uniform(*contrast_range)
    
    # Apply brightness & contrast
    img = slice_img + brightness_factor
    mean = np.mean(img)
    img = (img - mean) * contrast_factor + mean
    
    return img



def add_random_white_patches(slice_img, num_patches=10, max_patch_size=50):
    slice_mod = slice_img.copy()
    h, w = slice_mod.shape

    for _ in range(num_patches):
        patch_w = random.randint(10, max_patch_size)
        patch_h = random.randint(10, max_patch_size)
        x = random.randint(0, w - patch_w)
        y = random.randint(0, h - patch_h)

        slice_mod[y:y + patch_h, x:x + patch_w] = 900  
    return slice_mod
#-----------------------------------------------------------------------------------------------------------------------------

def preprocess_folder(png_folder) :
    # your file names contain dots - rename them to remove the dots for effective processing but keep the .png extension
    for file in os.listdir(png_folder):
        if file.endswith('.png'):
            base, ext = os.path.splitext(file)
            new_base = base.replace('.', '')
            new_file = new_base + ext
            old_path = os.path.join(png_folder, file)
            new_path = os.path.join(png_folder, new_file)
            # Only rename if the new filename doesn't already exist
            if old_path != new_path and not os.path.exists(new_path):
                os.rename(old_path, new_path)
                print(f'Renamed {file} to {new_file}')
            elif old_path != new_path:
                print(f'Skipped renaming {file} to {new_file} (target exists!)')


## convert all png files in nii files and save in a new folder called nii
def png_to_nii(folder_path, num_of_slices=256):
    preprocess_folder(folder_path)
    # create a new folder to save the nii files
    nii_folder = f"{folder_path}_nii_{num_of_slices}"
    if not os.path.exists(nii_folder):
        os.makedirs(nii_folder)
    # loop through all png files in the folder and convert them to nii files
    for file in os.listdir(folder_path):
        if file.endswith('.png'):
            img_path = os.path.join(folder_path, file)
            img = Image.open(img_path)
            img_np = np.array(img, dtype=np.float32)
            img_np = np.mean(img_np, axis=-1)[:,:,np.newaxis]

            # the model expects HU values (unit of brightness in a CT scan)
            # - the brightest intensity is bone (about 1500 HU) and the darkest intensity is air (about -1000 HU)
            intensity_max = 1500
            intensity_min = -1000 
            # take percentiles as max and min values for more robust normalization
            img_min = np.percentile(img_np, 2.5) 
            img_max = np.percentile(img_np, 97.5)
            img_np = np.clip(img_np, img_min, img_max)
            img_np = (img_np - img_min) / (img_max - img_min)
            img_np = (img_np * (intensity_max - intensity_min)) + intensity_min

            # rotate the image by 90 degrees (to match the orientation of the original data)
            img_np = np.rot90(img_np, 3)
         
            # subsample the image by a factor of 2 (to match the resolution of the original data)
            img_np = img_np[::2, ::2]

            # repeat the image by a factor of num_of_slices (to simulate 3D data)
            img_np = np.repeat(img_np, num_of_slices, axis=-1)
            for z in range(img_np.shape[-1]):
                if z % 2 == 1:
                    continue
                else:
                    sigma = abs((z - num_of_slices // 2) /num_of_slices) * 1.5
                    slice_mod = ndimage.gaussian_filter(img_np[:, :, z], sigma=sigma)
                    slice_mod = elastic_deformation(slice_mod, alpha=3, sigma=1)
                    slice_mod += np.random.normal(0, z % 2)
                    num_patches = (z % 6) + 1
                    max_patch_size = 11 + (z % 40)
                    slice_mod = add_random_white_patches(slice_mod, num_patches=num_patches, max_patch_size=max_patch_size)
                    slice_mod = direct_brightness_contrast(slice_mod, brightness_range=(0.9, 1.1), contrast_range=(0.9, 1.2))
                    img_np[:, :, z] = slice_mod

            # save the image
            nii_img = nib.Nifti1Image(img_np, affine=np.eye(4))
            nii_path = os.path.join(nii_folder, file.replace('.png', '.nii'))
            

            nii_img = nib.Nifti1Image(img_np, affine=np.eye(4))
            nii_path = os.path.join(nii_folder, file.replace('.png', '.nii'))
            nib.save(nii_img, nii_path)
    
    return nii_folder

