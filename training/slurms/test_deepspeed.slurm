#!/bin/bash -l
#SBATCH --clusters=genius
#SBATCH --account=intro_vsc36814
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=2
#SBATCH --partition=gpu_v100
#SBATCH --job-name=sft_test_qwen1B
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=seppe.vanswegenoven@student.kuleuven.be

echo "activating conda"
conda activate joketailor

echo "moving to training dir"
cd $VSC_DATA/JokeTailor/training

echo "starting training script"
accelerate launch --config_file configs/SFT_qwen32B_deepspeed_config.yaml start_training.py -f configs/SFT_qwen1B.json