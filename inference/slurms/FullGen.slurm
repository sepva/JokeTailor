#!/bin/bash -l
#SBATCH --clusters=wice
#SBATCH --account=lp_dtai1
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_a100
#SBATCH --job-name=FullGen
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=seppe.vanswegenoven@student.kuleuven.be

echo "activating conda"
conda activate joketailor

echo "moving to inference dir"
cd $VSC_DATA/JokeTailor/inference

echo "starting inference script"
python inference.py -f configs/FullGen_inference.json