#!/bin/bash

mkdir -p generations

SIZE=164
GPUS=8

pids=()
filename_prefix="santacoder_humaneval_generations"
file_names=""

for i in $(seq 0 $((GPUS-1)))
do
    ip=$(($i+1))
    string="Starting iteration $i with start and end  $((i*SIZE/GPUS)) $((ip*SIZE/GPUS))"
    echo $string
    CUDA_VISIBLE_DEVICES=$((i)) python main.py \
    --model bigcode/starcoder --use_auth_token \
    --trust_remote_code --tasks humaneval --batch_size 20 --n_samples 20 \
    --max_length_generation 1024 --precision bf16 --limit $SIZE \
    --save_generations --save_generations_path ./generations/${filename_prefix}_$((i*SIZE/GPUS)).json \
    --start $((i*SIZE/GPUS)) --end $((ip*SIZE/GPUS)) --shuffle &
    
    pids+=($!)

    file_names="$file_names ./generations/${filename_prefix}_$((i*SIZE/GPUS)).json"

done

echo "Spawned all processes with pids: "
echo ${pids[*]}

for pid in ${pids[*]}; do
    wait $pid
done

python combine_generations.py $file_names ./generations/${filename_prefix}.json
