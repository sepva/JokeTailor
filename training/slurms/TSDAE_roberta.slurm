#!/bin/bash -l
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc36814
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_h100
#SBATCH --job-name=TSDAE
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=seppe.vanswegenoven@student.kuleuven.be

echo "activating conda"
conda activate joketailor

echo "moving to training dir"
cd $VSC_DATA/JokeTailor/training

echo "starting training script"
python start_training.py -f configs/TSDAE.json