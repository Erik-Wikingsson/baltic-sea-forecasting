# Stage 1 of training a Graph-EFM model on Baltic Sea data
# "Auto-encoder" training

# The goal of this training is only for the model to learn to encode information
# into the latent variable, and decode this to feasible fields.
# The latent map (prior) is not trained here.
# We want the vi samples (predicted using variational distribution) to be as close
# to the trues states as possible.
# Since the prior isn't trained, the prior samples do not matter here.

# TO ADJUST: Constants below
BS=1
#CKPT= # Don't load checkpoint for step 1
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
    --kl_beta 0\
    --crps_weight 0\
    --ar_steps_train 1\
    --epochs 300\
    --val_interval 30\
    --ar_steps_eval 4\
    --val_steps_to_log 1 2 4\
    --var_leads_val_plot '{"10":[1,4], "7":[1,4], "1":[1,4]}'\
    #--load $CKPT\
