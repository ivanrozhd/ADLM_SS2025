#!/bin/bash
#SBATCH --job-name=ngocs-job 
#SBATCH --output=ngocs-job-%A.out
#SBATCH --error=ngocs-job-%A.err
#SBATCH --mail-user=bachngoc.doan@tum.de
#SBATCH --mail-type=ALL
##SBATCH --partition=universe,asteroids
##SBATCH --qos=master-queuesave
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Load python module
#ml python/anaconda3
source /meta/opt/anaconda3/etc/profile.d/conda.sh
# /u/home/doanb/.conda/envs/ngoc_adlm_env_py310

# Activate corresponding environment
# If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. 
# The following guards against that. Not necessary if you always run this script from a clean terminal
conda deactivate

# If the following does not work, try 'source activate <env-name>'
conda activate ngoc_adlm_env_py310

# Do in the future to store the environment
# conda env create -f environment.yml

# Run the program
python /vol/miltank/users/doanb/ADLM_SS2025/prompt-to-prompt-Stablediffusion/main_Prompt2MedImage.py