CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
  main_finetune.py \
  --batch_size 24 \
  --blr 2e-5 \
  --warmup_epochs 5 \
  --epochs 50 \
  --model spikformer_8_512_CAFormer \
  --data_path /raid/ligq/imagenet1-k \
  --output_dir outputs/55M_T4 \
  --log_dir outputs/55M_T4 \
  --model_mode ms \
  --dist_eval \
  --finetune checkpoint-299.pth \
  --time_steps 4 \
  --kd \
  --teacher_model caformer_b36_in21ft1k \
  --distillation_type hard
