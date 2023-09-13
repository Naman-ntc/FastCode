# Inference

We present the FastCode inference utilities here. The inference scripts are aligned with the bigcode-evaluation-harness to require minimal changes for people who are familiar with the harness.

**Note : Using vllm might have some regression in outputs. We have not obverved degradation in performance**

## Inference on a single GPU
To perform inference on a single GPU, use the following script
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--model bigcode/starcoder --use_auth_token \
--trust_remote_code --tasks humaneval --batch_size 20 --n_samples 20 \
--max_sequence_length 1024 --precision bf16 \
--save_generations --save_generations_path ./generations/starcoder_humaneval_generations.json
```
The inference loops over the dataset and generates `n_samples` completions for each problem using an inner loop determined by `batch_size`. Hence, for the best performance, `batch_size` should evenly divide the `n_samples`. 

## Distributed inference
Please look at `scripts` folder for optimized scripts. 
To perform distributed inference over the dataset, we will spawn individual processes for equal fraction of the dataset. The `--start {START}` and `--end {END}` flags provide functionality to access different portions of the dataset. Additionally, some datasets (like humaneval) are _semi-sorted_ by hardness which creates a bottleneck based on the _harder_ slice of the dataset. To avoid this, we shuffle the dataset before splitting using the `--shuffle` flag.

```bash
CUDA_VISIBLE_DEVICES=i python main.py \
--model bigcode/starcoder --use_auth_token \
--trust_remote_code --tasks humaneval --batch_size 20 --n_samples 20 \
--max_length_generation 1024 --precision bf16 \
--save_generations --save_generations_path ./generations/starcoder_humaneval_generations.json \
--start {START} --end {END} --shuffle
```

As mentioned above, for ideal performance `batch_size` should evenly divide the `n_samples`. 

## Performance
We have observed a good **3x** speedup using `vllm` for inference on `humaneval` (for the `starcoder` model with `max_sequence_length` equal to 1024). The difference is significantly more pronounced for longer sequences -- upto **20x** speedup for `max_sequence_length` equal to 2048 on the `APPS` dataset.