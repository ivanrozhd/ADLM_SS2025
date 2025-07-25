import nibabel as nib
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import os
import metrics_for_segmentation_masks as metrics

def extract_first_number(name):
    match = re.match(r"(\d+)", name)
    return str(match.group(1)) if match else float('inf')


def average_averaged_nifti_with_parent(folder1, folder2, save_root="results"):
    # TODO: Adjust - Extract "edit" or "orig" from ONE LEVEL UP
    parent_folder = os.path.basename(os.path.dirname(folder1))  # e.g., "nodules"
    edit_or_orig = "edit" if "edit" in os.path.basename(folder1) else "orig"

    #Create save directory structure: results/{parent}/{edit_or_orig}
    save_dir = os.path.join(save_root, parent_folder, edit_or_orig)
    os.makedirs(save_dir, exist_ok=True)

    #Get common subfolder IDs
    subfolders1 = {f for f in os.listdir(folder1) if os.path.isdir(os.path.join(folder1, f))}
    subfolders2 = {f for f in os.listdir(folder2) if os.path.isdir(os.path.join(folder2, f))}
    common_ids = sorted(list(subfolders1.intersection(subfolders2)))

    if not common_ids:
        print("No matching subfolder IDs found.")
        return

    for id_name in common_ids:
        sub1 = os.path.join(folder1, id_name)
        sub2 = os.path.join(folder2, id_name)

        files1 = {f for f in os.listdir(sub1) if f.endswith("_averaged.nii.gz")}
        files2 = {f for f in os.listdir(sub2) if f.endswith("_averaged.nii.gz")}
        common_files = sorted(list(files1.intersection(files2)))

        if not common_files:
            print(f"[{id_name}] No matching averaged files.")
            continue

        out_subdir = os.path.join(save_dir, id_name.replace("_seg", ""))
        os.makedirs(out_subdir, exist_ok=True)

        for file_name in common_files:
            mask1 = nib.load(os.path.join(sub1, file_name)).get_fdata()
            mask2 = nib.load(os.path.join(sub2, file_name)).get_fdata()
            affine = nib.load(os.path.join(sub1, file_name)).affine

            if mask1.shape != mask2.shape:
                print(f"[{id_name}] Skipping {file_name}: shape mismatch {mask1.shape} vs {mask2.shape}")
                continue

            averaged_mask = (mask1 + mask2) / 2.0

            averaged_nii_path = os.path.join(out_subdir, file_name.replace("_averaged", "_averaged_two"))
            nib.save(nib.Nifti1Image(averaged_mask, affine), averaged_nii_path)
            print(f"[{id_name}] Saved averaged mask: {averaged_nii_path}")

            avg_2d = np.mean(averaged_mask, axis=-1)
            averaged_png_path = averaged_nii_path.replace(".nii.gz", "_projection.png")
            plt.figure(figsize=(8, 8))
            plt.imshow(avg_2d.T, cmap="hot", origin="lower")
            plt.colorbar(label="Average mask intensity (two masks)")
            plt.title(f"Averaged Two Masks - {id_name} - {file_name}")
            plt.savefig(averaged_png_path, dpi=150)
            plt.close()
            print(f"[{id_name}] Saved heatmap: {averaged_png_path}")

    return  save_dir


def calculate_metrics(output_folder_original, output_folder_edited):
    results = []
    # List subfolders
    orig_subfolders = [f for f in os.listdir(output_folder_original) if os.path.isdir(os.path.join(output_folder_original, f))]
    edited_subfolders = [f for f in os.listdir(output_folder_edited) if os.path.isdir(os.path.join(output_folder_edited, f))]


    #Loop through each original subfolder
    for orig_sub in orig_subfolders:
        patient_id = extract_first_number(orig_sub)
        orig_sub_path = os.path.join(output_folder_original, orig_sub)

        # Find all edited subfolders for this patient
        matching_edited_subs = [f for f in edited_subfolders if f.startswith(patient_id)]

        for edited_sub in matching_edited_subs:
            edited_sub_path = os.path.join(output_folder_edited, edited_sub)

            # Loop through nii files inside the original folder
            for file_name in os.listdir(orig_sub_path):
                if not (file_name.endswith(".nii") or file_name.endswith(".nii.gz")):
                    continue

                orig_file = os.path.join(orig_sub_path, file_name)
                edited_file = os.path.join(edited_sub_path, file_name)

                if not os.path.exists(edited_file):
                    print(f"Missing in edited folder: {edited_file}")
                    continue

                # Load masks
                orig_mask = nib.load(orig_file).get_fdata() > 0
                edited_mask = nib.load(edited_file).get_fdata() > 0

                # If empty mask, mark corrupted
                if orig_mask.sum() == 0 or edited_mask.sum() == 0:
                    iou, dice = "corrupted", "corrupted"
                else:
                    iou = metrics.mask_iou(orig_mask, edited_mask)
                    dice = metrics.dice_coefficient(orig_mask, edited_mask)

                results.append({
                    "patient_id": patient_id,
                    "original_subfolder": orig_sub,
                    "edited_subfolder": edited_sub,
                    "structure": file_name,
                    "IoU": iou,
                    "Dice": dice
                })

    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("pneumonia_remove_segmentation_metrics.csv", index=False)
    print("\n Results saved!")
    return df