#!/bin/sh
#SBATCH -t 72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#SBATCH --output ./slurm_logs/%A_%x.out
#SBATCH -p berzelius-cpu -n1 -c32
#SBATCH -J GRAPH_10

cd /proj/berzelius-2022-164/users/x_erila/baltic-sea-forecasting

source ~/.bashrc
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate bsf
wandb online

python -m neural_lam.create_global_graph \
    --config_path /proj/berzelius-2022-164/weather/global_ocean/data_single/global_ocean_1_4.yaml \
    --name global_cluster_1_4_deg_10_refinement_4_levels \
    --type cluster \
    --plot \
    --levels 3 \
    --grid_to_first_mesh_refinement 10 \
    --mesh_refinement_factor 4 \
    --connect_disconnected