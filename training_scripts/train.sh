#!/bin/bash
#SBATCH -J FM_Overfit
#SBATCH -t 03-00:00:00
#SBATCH --gpus=1
#SBATCH -C "thin"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#

source ~/.bashrc
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate bsf
wandb online

cd /proj/berzelius-2022-164/users/x_erila/baltic-sea-forecasting

git switch main

bash training_scripts/graph_flow/graph_flow_1.sh
# bash training_scripts/graph_edm/graph_edm_1.sh