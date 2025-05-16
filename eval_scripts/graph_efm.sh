python -m neural_lam.train_model \
    --config_path tests/datastore_examples/mdp/danra_100m_winds/config.yaml \
    --model graph_efm\
    --n_example_pred 0\
    --graph hierarchical\
    --num_workers 16\
    --hidden_dim 128\
    --processor_layers 1\
    --prior_processor_layers 1\
    --encoder_processor_layers 1\
    --output_std \
    --ensemble_size 100\
    --batch_size 1\
    --eval test\
    # --load paper_checkpoints/graph_efm.ckpt\
