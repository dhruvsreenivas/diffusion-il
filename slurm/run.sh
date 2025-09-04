#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH -o /network/scratch/d/dhruv.sreenivas/diffusion-milo/job_logs/output/slurm-%j.out
#SBATCH -e /network/scratch/d/dhruv.sreenivas/diffusion-milo/job_logs/error/slurm-%j.err

# 1. Load your environment
module load anaconda/3
conda activate /home/mila/d/dhruv.sreenivas/anaconda3/envs/diffusion-milo

# 2. Launch the run.
python script/run.py $@