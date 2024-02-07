CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
  main_finetune.py \
  --batch_size 128 \
  --blr 6e-4 \
  --warmup_epochs 10 \
  --epochs 200 \
  --model spikformer_8_512_CAFormer \
  --data_path /raid/ligq/imagenet1-k \
  --output_dir outputs/55M \
  --log_dir outputs/55M \
  --model_mode ms \
  --dist_eval
