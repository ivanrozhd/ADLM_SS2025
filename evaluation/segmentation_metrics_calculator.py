import torch
import numpy as np
from colorama import Fore, Style
from scipy.spatial.distance import cdist
from skimage.metrics import hausdorff_distance
import cv2
from scipy.ndimage import label
from collections import defaultdict
import pandas as pd


def mask_iou(mask1, mask2):
    """ Input 
    mask1: numpy array of shape (H, W) with boolean values
    mask2: numpy array of shape (H, W) with boolean values
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou_score = intersection / union if union != 0 else 0.0
    return iou_score

def dice_coefficient(mask1, mask2):
    #mask1 = mask1.astype(bool)
    #mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    size1 = mask1.sum()
    size2 = mask2.sum()

    if size1 + size2 == 0:
        return 1.0  # Both masks are empty
    dice = 2. * intersection / (size1 + size2)
    return dice

def average_surface_distance(contour1, contour2):
    """
    contour1: numpy array of shape (N, 2) representing points in contour1
    contour2: numpy array of shape (M, 2) representing points in contour2
    """
    # Compute all pairwise distances between contour points
    dists_1_to_2 = cdist(contour1, contour2)
    dists_2_to_1 = cdist(contour2, contour1)

    # For each point in contour1, find distance to closest point in contour2
    min_dists_1_to_2 = dists_1_to_2.min(axis=1)
    # For each point in contour2, find distance to closest point in contour1
    min_dists_2_to_1 = dists_2_to_1.min(axis=1)

    # Average the minimal distances for both directions
    asd = (min_dists_1_to_2.mean() + min_dists_2_to_1.mean()) / 2.0
    return asd

# hd = hausdorff_distance(mask1, mask2)
def find_hd_points(u, v):
    # Distances from each u to closest v
    d_uv = cdist(u, v)
    u_to_v_min = d_uv.min(axis=1)
    max_u_idx = np.argmax(u_to_v_min)
    max_u_dist = u_to_v_min[max_u_idx]
    closest_v_idx_for_u = d_uv[max_u_idx].argmin()

    # Distances from each v to closest u
    d_vu = cdist(v, u)
    v_to_u_min = d_vu.min(axis=1)
    max_v_idx = np.argmax(v_to_u_min)
    max_v_dist = v_to_u_min[max_v_idx]
    closest_u_idx_for_v = d_vu[max_v_idx].argmin()

    # Decide which direction is the Hausdorff distance
    if max_u_dist > max_v_dist:
        return (u[max_u_idx], v[closest_v_idx_for_u]), max_u_dist
    else:
        return (u[closest_u_idx_for_v], v[max_v_idx]), max_v_dist
    
def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    Keeps only the largest connected component in a binary mask.
    """
    labeled_array, num_features = label(mask)
    if num_features == 0:
        return mask
    component_sizes = [(labeled_array == i).sum() for i in range(1, num_features + 1)]
    largest_label = np.argmax(component_sizes) + 1
    return (labeled_array == largest_label).astype(mask.dtype)

# Do 'Non-Maximum Suppression' for masks in an ultralytics.engine.results.Results
def mask_nms(mask_datas, plotting=False, iou_threshold=0.5):
    """
    mask_datas: torch.Tensor: ultralytics.engine.results.Results.Masks.data
    """

    kept_mask_data = []
    num_masks = len(mask_datas)
    suppressed = [False] * num_masks

    for i in range(num_masks):
        if suppressed[i]:
            continue
        mask_i = mask_datas[i].cpu().numpy()

        for j in range(i + 1, num_masks):
            if suppressed[j]:
                continue
            mask_j = mask_datas[j].cpu().numpy()

            if plotting:
                # plot the two mask next to each other
                """ Uncomment to plot in an interactive jupyter notebook
                plt.figure(figsize=(2, 2))
                plt.subplot(1, 2, 1)
                plt.imshow(mask_i, cmap="gray")
                plt.axis("off")
                plt.title("mask " + str(i))
                plt.subplot(1, 2, 2)
                plt.imshow(mask_j, cmap="gray")
                plt.axis("off")
                plt.title("mask " + str(j))
                plt.show() """

                print(f"IOU between mask {i} and mask {j} is {mask_iou(mask_i, mask_j)}")
            if mask_iou(mask_i, mask_j) > iou_threshold:
                # Suppress the smaller mask
                if mask_i.sum() >= mask_j.sum():
                    suppressed[j] = True
                else:
                    suppressed[i] = True
                    break  # no need to check others if i is suppressed

    for i in range(num_masks):
        if not suppressed[i]:
            kept_mask_data.append(mask_datas[i])

    return kept_mask_data

def mask_to_contours(mask: np.ndarray) -> list[np.ndarray]:
    mask_uint8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    contour_coords = [cnt.squeeze(axis=1) for cnt in contours if cnt.shape[0] > 2]
    return contour_coords


def evaluate_lung_masks(all_model_results, use_case, model_indices=[0, 1], iou_threshold=0.5, lung_mask_ratio=0.1, plot_hd=False, plotting=False):
    original_recognition_word = "original"

    # A structure matching the original to edited image results
    """seed_original_edited_results_map:
    {
        <seed>: {
            "original_result": {
                "img_name": <"....png">,
                "masks.data": torch.Tensor: ultralytics.engine.results.Results.masks.data
            },
            "edited_results": [
                {
                  "img_name": <"....png">,
                  "masks.data": torch.Tensor: ultralytics.engine.results.Results.masks.data
                },
                ...
            ]
        },
        ...
    }
    """
    # Temporary structure: seed -> img_name -> list of masks.data tensors
    temp_map = defaultdict(lambda: defaultdict(list))
    for model_results in [all_model_results[i] for i in model_indices]:
        #print(model_results["model_name"]) # SAM_l, or SAM2.1_b
        model_results = model_results["results"]
        #print(f"type of model_results: {type(model_results)}") # <class 'list'>
        #print(f"length of model_results: {len(model_results)}") # amount of images

        for result in model_results:
            result = result[0]  # ultralytics.engine.results.Results
            img_name = result.path.split('/')[-1]

            seed = int(img_name.split('_')[0])
            temp_map[seed][img_name].append(result.masks.data)

    # Final structured map
    seed_original_edited_results_map = {}
    for seed, img_dict in temp_map.items():
        seed_original_edited_results_map[seed] = {
            "original_result": None,
            "edited_results": []
        }

        for img_name, masks_list in img_dict.items():
            combined_masks = torch.cat(masks_list, dim=0)
            record = {
                "img_name": img_name,
                "masks.data": combined_masks
            }

            if (original_recognition_word in img_name) and (use_case in img_name):
                seed_original_edited_results_map[seed]["original_result"] = record
            else:
                seed_original_edited_results_map[seed]["edited_results"].append(record)



    metrics_rows = []

    for seed, result_pair in seed_original_edited_results_map.items():

        # Access original_result
        original_img_name = result_pair["original_result"]["img_name"]
        original_mask_data = result_pair["original_result"]["masks.data"]
        print(f"\nOriginal image: {original_img_name}")

        # For each detected mask, keep only largest connected component
        for mask_idx_1 in range(len(original_mask_data)):
            mask_np_1 = original_mask_data[mask_idx_1].cpu().numpy()
            mask_np_1 = keep_largest_connected_component(mask_np_1)
            original_mask_data[mask_idx_1] = torch.from_numpy(mask_np_1)

        # Do 'Non-Maximum Suppression' for original mask
        original_mask_data = mask_nms(original_mask_data, plotting=False, iou_threshold=iou_threshold)
        #print(f"Original image has {len(original_mask_data)} masks detected")

        # Access edited_results
        for edited in result_pair["edited_results"]:
            edited_img_name = edited["img_name"]
            edited_mask_data = edited["masks.data"]
            print(f"--- Edited image: {edited_img_name}")

            # For each detected mask, keep only largest connected component
            for mask_idx_2 in range(len(edited_mask_data)):
                mask_np_2 = edited_mask_data[mask_idx_2].cpu().numpy()
                mask_np_2 = keep_largest_connected_component(mask_np_2)
                edited_mask_data[mask_idx_2] = torch.from_numpy(mask_np_2)
            # Do 'Non-Maximum Suppression' for edited mask
            edited_mask_data = mask_nms(edited_mask_data, plotting=False, iou_threshold=iou_threshold)
            #print(f"Edited image has {len(edited_mask_data)} masks detected")

            # extract editing_configuration
            if (use_case not in edited_img_name) and (original_recognition_word in edited_img_name):
                editing_configuration = "No CrossAttn Control"
            else: # everything between "<seed>_" and "<use_case>" of edited_img_name
                editing_configuration = edited_img_name.split(f"{seed}_")[1].split(use_case)[0]
            #print(f"Editing configuration: {editing_configuration}")

            # calculate metrics for each pair of mask detected from original and edited
            for mask_idx_1 in range(len(original_mask_data)):
                mask_np_1 = original_mask_data[mask_idx_1].cpu().numpy()

                # Keep only big enough mask: > than lung_mask_ratio * image area (Cause we compare only lung lobes which are big in image)
                #print(f"Mask {mask_idx_1} is {round(mask_np_1.sum() / (mask_np_1.shape[0] * mask_np_1.shape[1]),3)} of whole image")
                if mask_np_1.sum() <= lung_mask_ratio * mask_np_1.shape[0] * mask_np_1.shape[1]:
                    continue

                # is_right/left_lung if the entire original mask is in the right/left 2/3 of the image
                H, W = mask_np_1.shape
                is_left_lung = False
                is_right_lung = False

                mask_1_x_indices = np.where(mask_np_1)[1]
                if mask_1_x_indices.size > 0:
                    min_x = mask_1_x_indices.min()
                    max_x = mask_1_x_indices.max()
                    one_third = W / 3

                    if max_x < one_third * 2:
                        is_left_lung = True
                    if min_x >= one_third:
                        is_right_lung = True

                for mask_idx_2 in range(len(edited_mask_data)):
                    mask_np_2 = edited_mask_data[mask_idx_2].cpu().numpy()

                    # Compare the masks only if their mask iou > iou_threshold
                    iou = round(mask_iou(mask_np_1, mask_np_2),2)
                    if iou > iou_threshold:

                        if is_left_lung ^ is_right_lung: #xor
                            if is_left_lung:
                                print(Fore.GREEN + 'LEFT LUNG' + Style.RESET_ALL)
                            else:
                                print(Fore.GREEN + 'RIGHT LUNG' + Style.RESET_ALL)

                            if plotting:
                                # plot the two mask side by side
                                """ Uncomment to plot in an interactive jupyter notebook
                                plt.figure(figsize=(2, 2))
                                plt.subplot(1, 2, 1)
                                plt.imshow(mask_np_1, cmap="gray")
                                plt.axis("off")
                                plt.title("mask " + str(mask_idx_1)) # title should be mask_idx_1 with
                                plt.subplot(1, 2, 2)
                                plt.imshow(mask_np_2, cmap="gray")
                                plt.axis("off")
                                plt.title("mask  " + str(mask_idx_2))
                                plt.show()
                                """ 

                            dice = round(dice_coefficient(mask_np_1, mask_np_2),2)
                            contour1 = mask_to_contours(mask_np_1)[0]
                            contour2 = mask_to_contours(mask_np_2)[0]
                            asd = round(average_surface_distance(contour1, contour2),2)
                            hd = hausdorff_distance(mask_np_1, mask_np_2)

                            if plot_hd:
                                # Plot contours and highlight the furthest points causing hd value
                                """ Uncomment to plot in an interactive jupyter notebook
                                (pointA, pointB), hd = find_hd_points(contour1, contour2) 
                                plt.figure(figsize=(3, 3))
                                plt.plot(contour1[:, 0], contour1[:, 1], label="Mask 1 contour", color='blue')
                                plt.plot(contour2[:, 0], contour2[:, 1], label="Mask 2 contour", color='green')
                                plt.scatter(*pointA, color='red', s=50, label="Max dist point on Mask 1")
                                plt.scatter(*pointB, color='orange', s=50, label="Closest point on Mask 2")
                                plt.gca().invert_yaxis()  # Invert y if image coordinates
                                plt.axis("off")
                                plt.show()
                                print()
                                """
                            hd = round(hd,2)

                            metrics_rows.append({
                                'Seed': seed,
                                'Editing Config': editing_configuration,
                                'Lung': 'left_lung' if is_left_lung else 'right_lung',
                                'IoU': iou,
                                'Dice': dice,
                                'ASD': asd,
                                'HD': hd
                            })

    return pd.DataFrame(metrics_rows)