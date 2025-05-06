#!/bin/bash -l
#SBATCH --clusters=genius
#SBATCH --account=lp_dtai1
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_p100
#SBATCH --job-name=bi_encoder
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=seppe.vanswegenoven@student.kuleuven.be

echo "activating conda"
conda activate joketailor

echo "moving to training dir"
cd $VSC_DATA/JokeTailor/training

echo "starting training script"
python start_training.py -f configs/BiEncoderTraining.json