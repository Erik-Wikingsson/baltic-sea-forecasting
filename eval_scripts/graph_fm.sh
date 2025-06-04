# python -m neural_lam.train_model \
#     --config_path tests/datastore_examples/mdp/danra_100m_winds/config.yaml \
#     --model graph_fm\
#     --n_example_pred 0\
#     --graph hierarchical\
#     --num_workers 16\
#     --hidden_dim 128\
#     --processor_layers 3\
#     --vertical_propnets 1\
#     --output_std \
#     --batch_size 1\
#     --load paper_checkpoints/graph_fm.ckpt\
#     --eval test\

python -m neural_lam.train_model \
  --config_path /proj/berzelius-2022-164/weather/neural_lam_datasets/baltic/data/baltic_nl_config_small.yaml \
  --model graph_fm \
  --graph hierarchical \
  --num_workers 1 \
  --epochs 20 \
  --batch_size 1 \
  --hidden_dim 4 \
  --processor_layers 2 \
  --num_past_forcing_steps 0 \
  --num_future_forcing_steps 0 \
  --ar_steps_eval 2
