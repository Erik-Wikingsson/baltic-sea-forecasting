# Stage 1 of training a Graph flow matching model on Baltic Sea data

# NOTE:
# Change the config_path to point to your dataset configuration file
# Change the --num_workers, --precision and --batch_size according to your system's capabilities

# TO ADJUST: Constants below
BS=1
#CKPT= # Don't load checkpoint for step 1
PREC=32 # Change to bf16 if suitable

srun python -m neural_lam.train_model \
    --config_path /proj/berzelius-2022-164/weather/neural_lam_datasets/baltic/10y-data/baltic_nl_config.yaml \
    --model FM \
    --precision $PREC\
    --n_example_pred 1 \
    --graph hierarchical \
    --num_workers 16 \
    --precision 32 \
    --hidden_dim 128 \
    --processor_layers 3 \
    --pred_residual \
    --vertical_propnets 1 \
    --num_past_forcing_steps 1 \
    --num_future_forcing_steps 1 \
    --ar_steps_train 1\
    --batch_size $BS \
    --epochs 600 \
    --ensemble_size 5\
    --val_interval 1\
    --ar_steps_eval 1\
    --val_steps_to_log 1\
    --var_leads_val_plot '{"10":[1], "7":[1], "1":[1]}'\
    --num_sanity_val_steps 1\
    # --eval test \
    # --load /proj/berzelius-2022-164/weather/neural_lam_datasets/baltic/10y-data/last.ckpt
