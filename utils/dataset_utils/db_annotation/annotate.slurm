#!/bin/bash -l
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc36814
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --ntasks=36
#SBATCH --gpus-per-node=2
#SBATCH --partition=gpu_a100
#SBATCH --job-name=annotation_test
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=seppe.vanswegenoven@student.kuleuven.be

echo "activating conda"
conda activate joketailor

echo "moving to annotate dir"
cd $VSC_DATA/JokeTailor/db_annotation

echo "starting annotation script (batch 3)"
python annotate_topics.py

