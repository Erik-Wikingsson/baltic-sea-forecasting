# Stage 3 of training a Graph EDM model on Baltic Sea data

# NOTE:
# Change the config_path to point to your dataset configuration file
# Change the --num_workers, --precision and --batch_size according to your system's capabilities
# Add the checkpoint path to the stage 1 model checkpoint

python -m neural_lam.train_model \
    --config_path data/baltic_nl_config_small.yaml \
    --model EDM \
    --n_example_pred 0 \
    --graph hierarchical \
    --num_workers 16 \
    --precision 32 \
    --hidden_dim 128 \
    --processor_layers 1 \
    --pred_redidual \
    --vertical_propnets 1 \
    --num_past_forcing_steps 1 \
    --num_future_forcing_steps 1 \
    --batch_size 1 \
    --epochs 1200 \
    --lr 0.00001 \
    --load checkpoint_path
