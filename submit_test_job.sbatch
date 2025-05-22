#!/bin/bash
#SBATCH --job-name=test-job
#SBATCH --output=test-job-%j.out
#SBATCH --error=test-job-%j.err
#SBATCH --mail-user=<your-email-address>
#SBATCH --mail-type=ALL
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=00:10:00  # Set the job time limit to 10 minutes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Load Python module
#module load python/anaconda3

# Activate the environment
conda deactivate

conda activate adlm  # Replace <env-name> with your environment name

# Run the test.py script
python ADLM_SS2025/model2.py
