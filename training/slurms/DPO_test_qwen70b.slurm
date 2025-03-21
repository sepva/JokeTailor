#!/bin/bash -l
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc36814
#SBATCH --nodes=2
#SBATCH --ntasks=36
#SBATCH --gpus-per-node=4
#SBATCH --partition=gpu_a100
#SBATCH --job-name=dpo_test_qwen
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=seppe.vanswegenoven@student.kuleuven.be

echo "activating conda"
conda activate joketailor

echo "moving to training dir"
cd $VSC_DATA/JokeTailor/training

echo "starting training script"
python start_training.py -f configs/DPO_qwen70B.json