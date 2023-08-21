# FastCode/FineTune
This folder provides efficient fine-tuning utilties for code generation models. [scripts/](scripts/) folder contains scripts for fine-tuning models on the code generation tasks for different models and different GPUs. We have optimized the training arguments for optimized performance for `V100`, `A6000`, and `A100` GPUs. We support efficient attention implementations for BigCode and LLaMA models which can be enabled by providing `--use_flash_attn` or `--use_xformer_attn` flags.

Following, we also provide fine-tuning performance benchmarks and analysis for the starcoder series of models. For each configuration, we measure the time per iteration with a batch size of 256 over a rack of 8 GPUs. First we measure the improvement provided by efficient attention implementations, provide optimal training configurations across gpus, and finally measure the effect of peft approaches.

## Effect of Attention Implementations
We compare the performance of the default attention implementation with xformer memory efficient attention and flash attention. We measure the time per iteration with a batch size of 256 on a rack of 8 A100s. Since, sequence length affects the optimal training arguments, we optimized training arguments for every attention implementation for every sequence length.
### Sequence Length - 1024
| Model | Attention Implementation | Time per Iteration |
| --- | --- | --- |
|starcoder-1b | Xformer Attention | 1.9 |
|starcoder-1b | Flash Attention | 1.75 |
|starcoder-1b | Default* | 3s |

<sup>*</sup>xformer and flash attention supported batch size of 4 while default attention only supported batch size 2.

### Sequence Length - 2048
| Model | Attention Implementation | Time per Iteration |
| --- | --- | --- |
|starcoder-1b | Xformer Attention | 4.0s |
|starcoder-1b | Flash Attention | 3.6s |
|starcoder-1b | Default* | 9.9s |
|starcoder-3b | Xformer Attention | 10.6s |
|starcoder-3b | Flash Attention | 9.5s |
|starcoder-3b | Default** | 21.1s |
|starcoder-7b | Xformer Attention | 25.5s |
|starcoder-7b | Flash Attention | 57s |
|starcoder-7b | Default*** | 91.3s |

<sup>*</sup>xformer and flash attention supported turning gradient checkpointing off while default attention required gradient checkpointing to be on.

<sup>**</sup>xformer and flash attention supported batch size of 8 while default attention only supported batch size 4 (FSDP enabled).

<sup>***</sup>xformer and flash attention supported batch size of 2 while default attention only supported batch size 1 (FSDP enabled).

Note that flash attention is underperforming for the larger 7b model. We will look into this and recommend using xformer in the meantime!
### Sequence Length - 4096
| Model | Attention Implementation | Time per Iteration |
| --- | --- | --- |
|starcoder-1b | Xformer Attention | TODO |
|starcoder-1b | Flash Attention | TODO |

## Effect of training configuration and GPUs
We evaluate the different training configurations (gradient checkpointing, batch size) across different GPUs here. The performance is measured using the xformer memory-efficient attention. Note that FSDP or deepspeed is not used since it is a 1B model. Note that the performance across GPUs is not directly configurable since they the respective machines are not identical. 

### V100 (32GB)
| Gradient Checkpointing | Gradient Accumulation Steps | Per Device Train Batch Size | Time per Iteration |
| --- | --- | --- | --- |
| True | 32 | 1 | 63.0s |

### A100 (40GB)
| Gradient Checkpointing | Gradient Accumulation Steps | Per Device Train Batch Size | Time per Iteration |
| --- | --- | --- | --- |
| True | 16 | 2 | 5.2s |
| False | 32 | 1 | 4.9s |
| False | 16 | 2 | 4.0s |

### A6000 (48GB)
| Gradient Checkpointing | Gradient Accumulation Steps | Per Device Train Batch Size | Time per Iteration |
| --- | --- | --- | --- |
| False | 32 | 1 | 20.3s |
| False | 16 | 2 | 19.3s |
| False | 8 | 4 | 18.3s |

## Effect of Peft Approaches
We use LoRA to fine-tune the santacoder model. We follow the same setup as above and measure time per iteration.

| Peft | Time per Iteration |
| --- | --- |
| None | 3.6 |
| LoRA (31% trainable parameters) | 3.1 |
