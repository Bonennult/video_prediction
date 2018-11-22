CUDA_VISIBLE_DEVICES=0 python scripts/train.py --input_dir data/bair --dataset bair \
  --model savp --model_hparams_dict hparams/bair_action_free/ours_savp/model_hparams.json \
  --output_dir logs/bair_action_free/ours_savp \
  --gpu_mem_frac 0.7 \
  --model_hparams tv_weight=0.001,transformation=flow
