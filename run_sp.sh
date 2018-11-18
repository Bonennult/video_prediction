CUDA_VISIBLE_DEVICES=0 python scripts/generate.py --input_dir data/bair \
  --dataset_hparams sequence_length=30 \
  --checkpoint pretrained_models/bair_action_free/ours_savp \
  --mode test \
  --results_dir results_test_samples/bair_action_free
