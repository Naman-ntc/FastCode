python main.py \
--model bigcode/gpt_bigcode-santacoder --use_auth_token \
--trust_remote_code --tasks humaneval --batch_size 20 --n_samples 20 \
--max_sequence_length 1024 --precision bf16 --temperature 0.3 \
--num_gpus 4 --exp_name santacoder_humaneval_03 --allow_code_execution --shuffle \