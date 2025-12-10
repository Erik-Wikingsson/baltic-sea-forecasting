# Stage 3 of training a Graph-EFM model on Baltic Sea data
# Unrolled ELBO training

# This is the same as the previous step, but unrolled over multiple steps.
# The same instructions apply, but keep in particular in mind that we want to
# see some diversity in prior samples as they are rolled out for longer time.

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
    --crps_weight 0\
    --ar_steps_train 4\
    --epochs 200\
    --val_interval 20\
    --ar_steps_eval 20\
    --val_steps_to_log 1 2 4 10 20\
    --var_leads_val_plot '{"10":[1,4,20], "7":[1,4,20], "1":[1,4,20]}'\
    --load $CKPT\
