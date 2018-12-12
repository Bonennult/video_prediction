CUDA_VISIBLE_DEVICES=0 python scripts/train.py --input_dir data/comma/train --dataset bair \
  --model savp --model_hparams_dict hparams/bair/ours_savp/model_hparams.json \
  --output_dir logs/comma_action/ours_svap \
  --model_hparams tv_weight=0.001,transformation=flow,downsample_layer=conv2d,upsample_layer=deconv2d,where_add=middle
