#!/bin/bash -l
#SBATCH --clusters=genius
#SBATCH --account=intro_vsc36814
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gpus-per-node=4
#SBATCH --partition=gpu_v100
#SBATCH --job-name=dpo_test_deepseekdistill14B_wo_bnb
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=seppe.vanswegenoven@student.kuleuven.be

echo "activating conda"
conda activate joketailor

echo "moving to training dir"
cd $VSC_DATA/JokeTailor/training

echo "starting training script"
ACCELERATE_LOG_LEVEL=info accelerate launch start_training.py -f configs/DPO_deepseekdistill14B.json

#