# Stage 1 of training a Graph EDM model on Baltic Sea data

# NOTE: 
# Change the config_path to point to your dataset configuration file
# Change the --num_workers and --batch_size according to your system's capabilities

python -m neural_lam.train_model \
    --config_path data/baltic_nl_config_small.yaml \
    --model EDM \
    --n_example_pred 0 \
    --graph hierarchical \
    --num_workers 16 \
    --hidden_dim 128 \
    --processor_layers 1 \
    --pred_redidual \
    --vertical_propnets 1 \
    --num_past_forcing_steps 2 \
    --num_future_forcing_steps 1 \
    --batch_size 1 \
    --epochs 600 \
