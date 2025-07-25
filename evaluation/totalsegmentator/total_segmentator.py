import os
import glob
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


def average_each_nifti(mask_dir):

    for filename in os.listdir(mask_dir):
        if not filename.endswith(".nii.gz"):
            continue

        mask_path = os.path.join(mask_dir, filename)
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        affine = mask_img.affine

        if np.max(mask_data) == 0:
            print(f"Skipping empty mask: {filename}")
            continue

        #Keep every second slice
        mask_data = mask_data[:, :, 1::2]

        #Average across all slices (2D heatmap)
        avg_2d = np.mean(mask_data, axis=-1)

        #Save averaged 2D heatmap as PNG
        png_path = os.path.join(mask_dir, filename.replace(".nii.gz", "_averaged.png"))
        plt.figure(figsize=(8, 8))
        plt.imshow(avg_2d.T, cmap="hot", origin="lower")
        plt.colorbar(label="Average mask intensity")
        plt.title(f"Averaged Mask - {filename}")
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"Saved heatmap: {png_path}")

        #Save averaged 2D as NIfTI (H, W, 1)
        avg_3d = avg_2d[:, :, np.newaxis]
        avg_nii_path = os.path.join(mask_dir, filename.replace(".nii.gz", "_averaged.nii.gz"))
        nib.save(nib.Nifti1Image(avg_3d, affine), avg_nii_path)
        print(f"Saved averaged NIfTI: {avg_nii_path}")

        # Optional: Thresholded binary mask
        binary_2d = (avg_2d > 0.2).astype(np.uint8)
        binary_3d = binary_2d[:, :, np.newaxis]
        binary_nii_path = os.path.join(mask_dir, filename.replace(".nii.gz", "_binary_thr0.2.nii.gz"))
        nib.save(nib.Nifti1Image(binary_3d, affine), binary_nii_path)
        print(f"Saved binary thresholded mask: {binary_nii_path}")



def segment(input_image_folder: str, output_file_path: str):

    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
    
    print(output_file_path)

    # Find all .nii files in the input folder
    nii_files = glob.glob(os.path.join(input_image_folder, '*.nii'))

    # loop through all nii files in the input folder and segment them
    for input_path in nii_files:
        # Get the base filename without extension
        base_name = os.path.basename(input_path)
        name_wo_ext = os.path.splitext(base_name)[0]
        output_path = os.path.join(output_file_path, f"{name_wo_ext}_seg")
        print(f"Processing {input_path} -> {output_path}")

        # apply different tasks
        #totalsegmentator(input_path, output_path,roi_subset=roi_subset_total) #heart
        #totalsegmentator(input_path, output_path,  task="lung_vessels")
        #totalsegmentator(input_path, output_path, task="pleural_pericard_effusion") - really bad
        totalsegmentator(input_path, output_path, task="lung_nodules")


        # also save the original nii image in the output folder
        shutil.copy(input_path, output_path)
        average_each_nifti(output_path)


