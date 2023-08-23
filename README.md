# FastCode

FastCode is a powerful set of utilities aimed to enhance the training, inference, and evaluation of code generation Large Language Models (LLMs). It is specifically designed to handle the challenges associated with code generation tasks and to streamline these processes, while offering a significant performance gain. Optimizations include efficient attention implementations, blazing-fast inference, and a simplified evaluation pipeline. 

FastCode integrates smoothly with prominent models such as [starcoder](https://huggingface.co/bigcode/starcoder) and [codegen2.5](https://huggingface.co/Salesforce/codegen25-7b-mono), and also provides a robust and simplified pipeline for evaluation using the bigcode-evaluation-harness! The training scripts have been benchmarked and configured for different devices (`V100`, `A6000` and `A100`).

## Setup and Installation
To clone the repository, run
```
git clone https://github.com/Naman-ntc/FastCode.git --recursive
```

For installations, we recommend using python version 3.8 or higher, CUDA version between 11.0 and 11.8 and GPUs with capability 7.0 or higher. To install, you can simply run `./setup.sh` which will install all the dependencies.

## FineTuning
FastCode supports [starcoder](https://huggingface.co/bigcode/starcoder) and [codegen2.5](https://huggingface.co/Salesforce/codegen25-7b-mono) family of models currently. FastCode finetuning is optimized using efficient attention implementations like flash-attention and memory-efficient-attention. FastCode provides over **3x** speed up for larger models and over **2x** speed up for smaller models. `finetune/README` contains more details about the training arguments and the performance benchmarks for different models and different GPUs. `finetune/scripts` contains some example scripts for fine-tuning models on the code generation tasks for different models and different GPUs. 

## Inference
Evaluating code generation models is compute intensive since model is evaluated over multiple generations. Specifically, `pass@k` metric requires multiple samples per example (`n_samples`) and can be as large as 200. For example, performing inference over the entire humaneval dataset (164 problems) (with `n_samples` = 20) takes about 25 minutes on 8 A100 (40 GB) GPUs. Therefore, we need extremely fast inference support to iterate on results quickly.

FastCode uses [`vllm`](https://vllm.readthedocs.io/en/latest/index.html) for blazing fast inference in this repository! This speeds up the inference by **3 times** for smaller sequence lengths (1024 used for humaneval) and over **30 times** for longer sequence lengths (2048 used for apps). The `inference/scripts` folder depicts examples for performing inference on humaneval dataset for `starcoder` and `santacoder` models.

## Evaluation
FastCode uses [`bigcode-evaluation-harness`](https://github.com/bigcode-project/bigcode-evaluation-harness) to perform evaluation over the generated samples. The inference step above generates the data in the appropriate format as expected by the harness. To perform evaluation on the humaneval dataset follow the script below

```
cd evaluation/bigcode-evaluation-harness
python main.py --tasks humaneval --allow_code_execution --n_samples {n} --limit {limit} --load_generations_path {path}
```

## Roadmap 

- [x] Improve inference scripts further to use GPUs more efficiently
- [x] Add lora finetuning support
- [x] Add finetuning performance benchmarks for attention and lora
- [x] Support FlashAttention for BigCode models
- [ ] Add finetuning performance benchmarks for larger models
- [ ] Add quantized model training support
- [ ] Add quantized model inference support

## Citation
If you find this repository useful, please cite this as
```
@misc{fastcode,
  author = {Naman Jain},
  title = {FastCode: Utilities for better training, inference and evaluation of code generation models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Naman-ntc/FastCode}},
}
```


