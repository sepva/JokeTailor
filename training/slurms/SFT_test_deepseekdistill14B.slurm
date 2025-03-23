#!/bin/bash -l
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc36814
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --gpus-per-node=4
#SBATCH --partition=gpu_a100
#SBATCH --job-name=sft_test_deepseekdistill14B_wo_bnb
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=seppe.vanswegenoven@student.kuleuven.be

echo "activating conda"
conda activate joketailor

echo "moving to training dir"
cd $VSC_DATA/JokeTailor/training

echo "starting training script"
python start_training.py -f configs/SFT_deepseekdistill14B_higher_batch.json

#ACCELERATE_LOG_LEVEL=info accelerate launch