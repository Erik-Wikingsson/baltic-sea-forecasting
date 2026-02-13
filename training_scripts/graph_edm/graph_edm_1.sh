# Stage 1 of training a Graph EDM model on Baltic Sea data

# NOTE:
# Change the config_path to point to your dataset configuration file
# Change the --num_workers, --precision and --batch_size according to your system's capabilities

python -m neural_lam.train_model \
    --config_path /proj/berzelius-2022-164/weather/neural_lam_datasets/baltic/graph_rework_data/baltic_graph_rework.yaml \
    --model EDM \
    --precision bf16-mixed \
    --n_example_pred 1 \
    --graph hierarchical \
    --num_workers 12 \
    --precision bf16-mixed \
    --hidden_dim 64 \
    --processor_layers 2 \
    --pred_residual \
    --num_past_forcing_steps 1 \
    --num_future_forcing_steps 1 \
    --ar_steps_train 1 \
    --batch_size 6 \
    --epochs 600 \
    --lr 0.001 \
    --ensemble_size 5 \
    --val_interval 30 \
    --ar_steps_eval 4 \
    --num_sanity_val_steps 0 \
    --val_steps_to_log 1 2 4 \
    --var_leads_val_plot '{"0":[1,4], "1":[1,4], "2":[1,4], "3":[1,4], "4":[1,4]}' \
    # --subset_ds \
    # --devices 3 4 5
