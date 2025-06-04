python -m neural_lam.train_model \
    --config_path /proj/berzelius-2022-164/weather/neural_lam_datasets/baltic/data/baltic_nl_config_small.yaml \
    --model graph_efm\
    --n_example_pred 0\
    --graph hierarchical\
    --num_workers 1\
    --hidden_dim 128\
    --processor_layers 1\
    --prior_processor_layers 1\
    --encoder_processor_layers 1\
    --ensemble_size 100\
    --batch_size 1\
    --ar_steps_eval 2 # We only have 6 steps in the small dataset
    # --output_std \
    # --eval test\
    # --load paper_checkpoints/graph_efm.ckpt\
