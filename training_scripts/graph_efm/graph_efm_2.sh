# Stage 2 of training a Graph-EFM model on Baltic Sea data
# 1-step ELBO training

# The goal of this training is to learn to capture the distribution at one step.
# Here all parts of the model is trained.
# For single time step prediction, we want the model here to learn to produce
# plausible fields both using samples from the variational and prior distribution.
# We also want to see some diversity in the samples when using the prior.
# If the model collapses to deterministic forecasts => reduce kl_beta
# If the model does not produce reasonable-looking samples from the prior
# => increase kl_beta
# However, the spread-skill-ratio is not crucial to be ~1 here, as that will be
# helped by the CRPS in the last step.

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
    --lr 0.001\
    --kl_beta 1\
    --crps_weight 0\
    --ar_steps_train 1\
    --epochs 100\
    --val_interval 20\
    --ar_steps_eval 4\
    --val_steps_to_log 1 2 4\
    --var_leads_val_plot '{"10":[1,4], "7":[1,4], "1":[1,4]}'\
    --load $CKPT\
