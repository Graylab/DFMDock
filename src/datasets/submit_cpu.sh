#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --partition=parallel
#SBATCH --account=jgray21
#SBATCH --time=12:00:00
#SBATCH --output=slogs/%j.out

#### execute code
python docking_dataset.py
