#!/bin/bash -l
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc36814
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_a100_debug
#SBATCH --job-name=BoN
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=seppe.vanswegenoven@student.kuleuven.be

echo "activating conda"
conda activate joketailor

echo "moving to inference dir"
cd $VSC_DATA/JokeTailor/inference

echo "starting inference script"
python inference.py -f configs/BoN_gen_test.json