python -m neural_lam.train_model \
    --config_path data/baltic_nl_config_small.yaml \
    --model EDM \
    --graph hierarchical \
    --num_workers 2 \
    --hidden_dim 128 \
    --processor_layers 3 \
    --num_past_forcing_steps 1 \
    --num_future_forcing_steps 1 \
    --ensemble_size 5 \
    --batch_size 4 \
    --epochs 200 \
    --num_sanity_val_steps 0 \
    --val_interval 10\
    --ar_steps_eval 1\
    --val_steps_to_log 1\
    --var_leads_val_plot '{"10":[1,4], "7":[1,4], "1":[1,4]}'\
    --n_example_pred 1 \
    --ar_steps_eval 1\
    # --eval test \
    # --load /proj/berzelius-2022-164/weather/neural_lam_datasets/baltic/10y-data/last.ckpt
    # --output_std \
