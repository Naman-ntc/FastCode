#!/bin/bash

mkdir -p generations

SIZE=164
GPUS=8

pids=()

for i in $(seq 0 $((GPUS-1)))
do
    ip=$(($i+1))
    string="Starting iteration $i with start and end  $((i*SIZE/GPUS)) $((ip*SIZE/GPUS))"
    echo $string
    CUDA_VISIBLE_DEVICES=$((i)) python main.py \
    --model bigcode/gpt_bigcode-santacoder --use_auth_token \
    --trust_remote_code --tasks humaneval --batch_size 20 --n_samples 20 \
    --max_length_generation 1024 --precision bf16 \
    --save_generations --save_generations_path ./generations/santacoder_humaneval_generations_$((i*SIZE/GPUS)).json \
    --start $((i*SIZE/GPUS)) --end $((ip*SIZE/GPUS)) --randomize &
    
    pids+=($!)

    # --save_references --save_references_path ./generations/santacoder_humaneval_references_$((i*SIZE/GPUS)).json \

done

for pid in ${pids[*]}; do
    wait $pid
done

file_names=""
for i in $(seq 0 $((GPUS-1)))
do
    file_names="$file_names ./generations/santacoder_humaneval_generations_$((i*SIZE/GPUS)).json"
done

python combine_generations.py $file_names ./generations/santacoder_humaneval_generations.json

# file_names=""
# for i in $(seq 0 $((GPUS-1)))
# do
#     file_names="$file_names ./generations/santacoder_humaneval_references_$((i*SIZE/GPUS)).json"
# done

# python combine_references.py $file_names ./generations/santacoder_humaneval_references.json