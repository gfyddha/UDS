#! /bin/bash

cd ..

CUDA_VISIBLE_DEVICES=0 python finetune.py \
         --base_model "/path/to/model/Qwen2.5-7B" \
         --data_path "/path/to/mmlu" \
         --output_dir "/path/to/output_dir" \
         --batch_size 128 \
         --micro_batch_size 4 \
         --num_epochs 1 \
         --learning_rate 1.5e-4 \
         --cutoff_len 512 \
         --val_set_size 0 \
         --lora_r 8 \
         --lora_alpha 16 \
         --warmup_rates 0.01 \
         --lora_dropout 0.00 \
         --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
         --train_on_inputs \
         --group_by_length \
         --add_eos_token \
         --prompt_template_name "alpaca_mmlu" \
         --A_type "gaussian" \
         --start_sampling_step 0 \
         --seed 21 \
         --queue_size 1024 \
         --projection_dim1 128 \
         --projection_dim2 8 \
         --alpha 1.5e-3 \
         --k_base 1 \
         --output_file "train_mmlu"

CUDA_VISIBLE_DEVICES=0 python evaluate.py \
         --base_model "/path/to/model/Qwen2.5-7B" \
         --lora_weights "/path/to/output_dir" \
         --test_dataset "/path/to/mmlu" \
         --batch_size 8 \
         --prompt_template "alpaca_mmlu" \
         --max_new_tokens 128 \
         --save_path "/path/to/output_dir/lora_test_mmlu.json" \
         --seed 21