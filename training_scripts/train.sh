#!/bin/sh
#SBATCH -t 72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#SBATCH --output ./slurm_logs/%A_%x.out
#SBATCH -J SeaCast
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --gpus-per-node=8
# --dependency=afterany:15131
# NOTE: Makes this job run after the job with ID JOB_ID completes, regardless of its exit status.

source ~/.bashrc
module load Miniforge3/25.3.1-0
mamba activate bsf

cd /proj/berzelius-2022-164/users/x_erila/baltic-sea-forecasting
wandb online

# bash training_scripts/graph_flow/graph_flow_1.sh
# bash training_scripts/graph_edm/graph_edm_1.sh
# bash training_scripts/graph_crps/graph_crps_1.sh
# bash training_scripts/graph_SI/graph_SI_1.sh
# bash training_scripts/graph_fm/graph_fm_pretrain.sh
# bash training_scripts/seacast/seacast_pretrain.sh
bash training_scripts/seacast/seacast_pretrain_muon.sh