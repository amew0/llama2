#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=circle_area
#SBATCH --time=02:00:00
#SBATCH --partition=prod
#SBATCH --account=kunf0007
#SBATCH --output=circle_area.%j.out
#SBATCH --error=circle_area.%j.err
 
module purge
module load miniconda/3
echo "started" 
python circle_area.py 5