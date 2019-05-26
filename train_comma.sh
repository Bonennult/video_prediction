CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --input_dir data/comma_m/train --dataset bair \
  --model savp --model_hparams_dict hparams/bair_action_free/ours_savp/model_hparams.json \
  --output_dir logs/comma_m_1903/ours_svap \
