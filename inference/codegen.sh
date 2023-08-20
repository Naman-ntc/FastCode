#!/bin/bash

mkdir -p generations

SIZE=702
GPUS=8

pids=()
filename_prefix="codegen_plan_generations"
file_names=""

for i in $(seq 0 $((GPUS-1)))
do
    ip=$(($i+1))
    string="Starting iteration $i with start and end  $((i*SIZE/GPUS)) $((ip*SIZE/GPUS))"
    echo $string
    CUDA_VISIBLE_DEVICES=$((i)) python main.py \
    --model ../finetune/checkpoints_codegen25_small_data_ebs_256_lr_5e5_ep8/ --use_auth_token \
    --trust_remote_code --tasks apps-introductory-cfstyle --batch_size 50 --n_samples 200 \
    --max_length_generation 2048 --precision bf16 --limit $SIZE --temperature 0.8 \
    --save_generations --save_generations_path ./generations/$filename_prefix_$((i*SIZE/GPUS)).json \
    --start $((i*SIZE/GPUS)) --end $((ip*SIZE/GPUS)) --shuffle &
    
    pids+=($!)

    file_names="$file_names ./generations/$filename_prefix_$((i*SIZE/GPUS)).json"
done

echo "Spawned all processes with pids: "
echo ${pids[*]}

for pid in ${pids[*]}; do
    wait $pid
done

python combine_generations.py $file_names ./generations/$filename_prefix.json