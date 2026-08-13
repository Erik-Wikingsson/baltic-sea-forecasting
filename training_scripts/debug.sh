wandb off

# python -m neural_lam.train_model \
#     --config_path /proj/berzelius-2022-164/weather/global_ocean/data_single/global_ocean_1_4.yaml \
#     --num_workers 4 \
#     --precision bf16-mixed \
#     --model graph_efm \
#     --graph global_cluster_1_4_deg_10_refinement_4_levels \
#     --hidden_dim 672 \
#     --hidden_dim_mesh_nodes 128 \
#     --hidden_dim_grid 128 \
#     --hidden_dim_edge 32 \
#     --processor_layers 6 \
#     --n_example_pred 1 \
#     --batch_size 1 \
#     --epochs 100 \
#     --lr 0.001 \
#     --kl_beta 0.1 \
#     --ar_steps_train 1 \
#     --val_interval 25 \
#     --ar_steps_eval 1 \
#     --num_sanity_val_steps 0 \
#     --val_steps_to_log 1 \
#     --var_leads_val_plot '{"0":[1,4], "1":[1,4], "2":[1,4], "3":[1,4], "9":[1,4], "15":[1,4], "21":[1,4]}' \
#     --num_nodes 1
    # --crps_weight 1e-6 \
    # --grad_checkpointing \


python -m neural_lam.train_model \
    --config_path data/global_ocean_1_density.yaml \
    --num_workers 2 \
    --precision bf16-mixed \
    --model graph_det \
    --graph global_cluster_1_deg_20_refinement_4_levels \
    --hidden_dim 256 \
    --hidden_dim_grid 128 \
    --hidden_dim_mesh_nodes 128 \
    --hidden_dim_edge 32 \
    --processor_layers 3 \
    --input_steps 2 \
    --batch_size 1 \
    --lr 0.001 \
    --scheduler deterministic \
    --val_interval 5 \
    --ar_steps_eval 4 \
    --val_steps_to_log 1 4 \
    --num_sanity_val_steps 1 \
    --num_nodes 1 \
    --optimizer "muon_flat" 