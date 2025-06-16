# Stage 1 of training a Graph-EFM model on Baltic Sea data

# TO ADJUST: Constants below
BS=1
#CKPT= #Don't load checkpoint for step 1

python -m neural_lam.train_model \
    --config_path data/baltic_nl_config_small.yaml \
    --num_workers 4\
    --model graph_efm\
    --graph hierarchical\
    --hidden_dim 128\
    --processor_layers 1\
    --prior_processor_layers 1\
    --encoder_processor_layers 1\
    --num_past_forcing_steps 1 \
    --num_future_forcing_steps 1 \
    --n_example_pred 1\
    --ensemble_size 5\
    --batch_size $BS\
    --kl_beta 0\
    --crps_weight 0\
    --ar_steps_train 1\
    --epochs 300\
    --ar_steps_eval 4\
    --val_steps_to_log 1 2 4\
    --var_leads_val_plot "{0:[TODO], 1:[TODO]}"\
    #--load $CKPT\
