python sft_paddleocr_vl.py \
    --run_name "PaddleOCR-VL-For-Manga" \
    --model_path /home/PaddleOCR-VL \
    --use_flash_attention_2 \
    --split train \
    --max_length 2048 \
    --pad_to_multiple_of 8 \
    --output_dir ../sft_output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 5000 \
    --save_total_limit 3 \
    --bf16 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory \
    --gradient_checkpointing \
    --optim adamw_torch_fused \
#    --resume_from_checkpoint ../sft_output/checkpoint-10000 \
#    --report_to none
    --report_to wandb  # Options: none, wandb, tensorboard, or "wandb tensorboard" for both
