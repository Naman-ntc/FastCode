torchrun --nproc_per_node=8 --rdzv-endpoint localhost:29512 fastcode_monkeypatch_trainer.py \
    --model_name_or_path Salesforce/codegen25-7b-mono \
    --low_cpu_mem_usage True \
    --use_xformer_attn True \
    --bf16 True \
    --tf32 True \
    --output_dir checkpoints_codegen25_small_data_ebs_256_lr_5e5 \
    --num_train_epochs 16 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 32 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --learning_rate 5e-5 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --block_size 2048 \
    --report_to wandb \
    --run_name fastcode_codegen25_no_fn_ebs_256_lr_5e-5_ep_16 \
    --do_train \
    --do_eval \
    --deepspeed ds_config_zero3.json \
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
