# Stage 4 of training a Graph-EFM model on Baltic Sea data
# Unrolled training ELBO + CRPS

# This is the final fine-tuning, with the added CRPS loss.
# This is really the first time the latent map is trained in a roll-out setting.
# It should help both with calibrating the spread (closer to
# spread-skill-ratio = 1) and with reducing RMSE over longer lead times.
# If no improvement is seen for the things mentioned above
# => increase --crps_weight (by a factor x10)
# If spatially incoherent artefacts start appearing => decrease --crps_weight

# TO ADJUST: Constants below
BS=1
CKPT=path_to.ckpt
PREC=32 # Change to bf16 if suitable

python -m neural_lam.train_model \
    --config_path data/baltic_nl_config_small.yaml\
    --num_workers 4\
    --precision $PREC\
    --model graph_efm\
    --graph hierarchical\
    --hidden_dim 128\
    --processor_layers 1\
    --prior_processor_layers 1\
    --encoder_processor_layers 1\
    --num_past_forcing_steps 1\
    --num_future_forcing_steps 1\
    --n_example_pred 1\
    --ensemble_size 5\
    --batch_size $BS\
    --lr 0.0005\
    --kl_beta 1\
    --crps_weight 0.00001\
    --ar_steps_train 4\
    --epochs 100\
    --val_interval 20\
    --ar_steps_eval 20\
    --val_steps_to_log 1 2 4 10 20\
    --var_leads_val_plot '{"10":[1,4,20], "7":[1,4,20], "1":[1,4,20]}'\
    --load $CKPT\
