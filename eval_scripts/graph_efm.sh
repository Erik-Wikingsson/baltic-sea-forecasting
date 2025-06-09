python -m neural_lam.train_model \
    --config_path data/baltic_nl_config_small.yaml \
    --model graph_efm\
    --n_example_pred 0\
    --graph hierarchical\
    --num_workers 2\
    --hidden_dim 128\
    --processor_layers 1\
    --prior_processor_layers 1\
    --encoder_processor_layers 1\
    --ensemble_size 5\
    --batch_size 1\
    --num_past_forcing_steps 0 \
    --num_future_forcing_steps 0 \
    --ar_steps_eval 5 \
    --eval test\
    --n_example_pred 1 \
    # --output_std \
    # --load paper_checkpoints/graph_efm.ckpt\
