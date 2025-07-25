import total_segmentator
import png_to_nii as converter
import utils
from plot import plot_metrics_with_config_extraction
import os
import nibabel as nib
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches



if __name__ == "__main__": 

    #Define ur path for evaluation
    png_folder = '/vol/miltank/users/rozhdest/totalsegmentator/data/pneumonia_remove' 
    use_case = "pneumonia_add_remove"

    png_folder_original = os.path.join(png_folder, "orig")

    png_folder_edited = os.path.join(png_folder, "edit")

    if os.path.exists(png_folder_original):
        print("Found:", png_folder_original)
    else:
        print("No 'orig' subdirectory found")
        
    if os.path.exists(png_folder_edited):
        print("Found:", png_folder_edited)
    else:
        print("No 'orig' subdirectory found")

    # convert images to nii
    nii_folder_original_128 = converter.png_to_nii(png_folder_original,128)
    nii_folder_original_256 = converter.png_to_nii(png_folder_original,256)
    nii_folder_edited_128 = converter.png_to_nii(png_folder_edited,128)
    nii_folder_edited_256 = converter.png_to_nii(png_folder_edited,256)

    # create an output folder
    base_folder_name_original_128 = os.path.basename(nii_folder_original_128)
    parent_folder_original_128 = os.path.dirname(nii_folder_original_128)
    output_folder_original_128 = os.path.join(parent_folder_original_128, base_folder_name_original_128 + "_segmented")
    base_folder_name_original_256 = os.path.basename(nii_folder_original_256)
    parent_folder_original_256 = os.path.dirname(nii_folder_original_256)
    output_folder_original_256 = os.path.join(parent_folder_original_256, base_folder_name_original_256 + "_segmented")

    base_folder_name_edited_128 = os.path.basename(nii_folder_edited_128)
    parent_folder_edited_128 = os.path.dirname(nii_folder_edited_128)
    output_folder_edited_128 = os.path.join(parent_folder_edited_128, base_folder_name_edited_128 + "_segmented")
    base_folder_name_edited_256 = os.path.basename(nii_folder_edited_256)
    parent_folder_edited_256 = os.path.dirname(nii_folder_edited_256)
    output_folder_edited_256 = os.path.join(parent_folder_edited_256, base_folder_name_edited_256 + "_segmented")


    # create masks
    total_segmentator.segment(nii_folder_original_128, output_folder_original_128)
    total_segmentator.segment(nii_folder_original_256, output_folder_original_256)
    total_segmentator.segment(nii_folder_edited_128,output_folder_edited_128)
    total_segmentator.segment(nii_folder_edited_256,output_folder_edited_256)

    
    # take the average of the masks
    output_folder_original = utils.average_averaged_nifti_with_parent(output_folder_original_128, output_folder_original_256)
    output_folder_edited = utils.average_averaged_nifti_with_parent(output_folder_edited_128, output_folder_edited_256)

    avg_per_seed_config_metrics_df = utils.calculate_metrics(output_folder_original, output_folder_edited)
    df_filtered = avg_per_seed_config_metrics_df[avg_per_seed_config_metrics_df["structure"] != "lung_nodules_averaged_two.nii.gz"]


    # Plotting
    metric = 'Dice'
    title = 'Dice score across seeds'
    y_label = 'Dice score'
    legend_loc = 'lower right'
    x_label = 'crossReplace_selfReplace-steps'
    config_order = ['0.8_0.6', '0.5_0.6', '0.2_0.6', '0.8_0.4', '0.5_0.4', '0.2_0.4', '0.8_0.2', '0.5_0.2', '0.2_0.2', '0.0_0.0']
    
   
    plot_metrics_with_config_extraction(df=df_filtered, 
                 metric=metric, 
                 title=title,
                 config_order=config_order, 
                 x_label=x_label, 
                 y_label=y_label, 
                 legend_loc=legend_loc, 
                 output_folder='results/' + use_case)


