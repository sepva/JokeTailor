#!/bin/bash -l
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc36814
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_h100
#SBATCH --job-name=FullMinusBoN
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=seppe.vanswegenoven@student.kuleuven.be

echo "activating conda"
conda activate joketailor

echo "moving to inference dir"
cd $VSC_DATA/JokeTailor/inference

echo "starting inference script"
python inference.py -f configs/FullMinusBoN.json