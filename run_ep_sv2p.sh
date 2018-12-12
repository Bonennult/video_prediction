CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --input_dir data/comma \
  --dataset sv2p \
  --dataset_hparams dataset_name='humans' \
  --model sv2p \
  --mode test \
  --results_dir results_test/comma \
  --batch_size 1 \
  --num_stochastic_samples 10 \
  --gpu_mem_frac 0.7 \
  --eval_parallel_iterations 1
