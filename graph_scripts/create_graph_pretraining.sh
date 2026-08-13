#!/bin/sh
#SBATCH -t 72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#SBATCH --output ./slurm_logs/%A_%x.out
#SBATCH -p berzelius-cpu -n1 -c32
#SBATCH -J GRAPH_20

cd /proj/berzelius-2022-164/users/x_erila/baltic-sea-forecasting

source ~/.bashrc
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate bsf
wandb online

# global_cluster_1_4_deg_10_refinement_4_levels
# global_cluster_1_4_deg_15_refinement_4_levels
# global_cluster_1_4_deg_20_refinement_4_levels


python -m neural_lam.create_auxiliary_graph \
    --config_path /proj/berzelius-2022-164/weather/global_ocean/data/global_ocean_1.yaml \
    --name global_cluster_1_deg_20_refinement_4_levels \
    --name_original global_cluster_1_4_deg_20_refinement_4_levels \
    --plot \
    --connect_disconnected