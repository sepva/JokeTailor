#!/bin/bash -l
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc36814
#SBATCH --nodes=1
#SBATCH --ntasks=36
#SBATCH --gpus-per-node=2
#SBATCH --partition=gpu_a100
#SBATCH --time=01:00:00
#SBATCH --job-name=sft_qwen32_deepspeed
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=seppe.vanswegenoven@student.kuleuven.be

echo "activating conda"
conda activate joketailor

echo "moving to training dir"
cd $VSC_DATA/JokeTailor/training

echo "starting training script"
accelerate launch --config_file configs/SFT_qwen32B_deepspeed_config.yaml start_training.py -f configs/SFT_qwen32B.json

#
#accelerate launch --config_file configs/test_config.yaml