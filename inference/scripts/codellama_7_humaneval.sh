python main.py \
--model codellama/CodeLlama-7b-hf --use_auth_token \
--trust_remote_code --tasks humaneval --batch_size 10 --n_samples 10 \
--max_sequence_length 1024 --precision bf16 --temperature 0.3 \
--num_gpus 4 --exp_name codellama7_humaneval_03 --allow_code_execution --shuffle \