# FastCode/FineTune
This folder provides efficient fine-tuning utilties for code generation models. [scripts/](scripts/) folder contains scripts for fine-tuning models on the code generation tasks for different models and different GPUs. We have optimized the training arguments for optimized performance for `V100`, `A6000`, and `A100` GPUs. We support efficient attention implementations for BigCode and LLaMA models which can be enabled by providing `--use_flash_attn` or `--use_xformer_attn` flags.

Following, we also provide fine-tuning performance benchmarks and analysis for the `santacoder` (starcoder-1b) model. For each configuration, we measure the time per iteration with a batch size of 256 over a rack of 8 GPUs. First we measure optimal training configurations across gpus and then compare the effect of attention implementations.


## Effect of training configuration and GPUs
We evaluate the different training configurations (gradient checkpointing, batch size) across different GPUs here. The performance is measured using the xformer memory-efficient attention. Note that FSDP or deepspeed is not used since it is a 1B model.
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

## Effect of Attention Implementations
We compare the performance of the default attention implementation with xformer memory efficient attention and flash attention. We measure the time per iteration with a batch size of 256. Since, sequence length affects the optimal training arguments, we optimized training arguments for every attention implementation for every sequence length.
### Sequence Length - 1024
| Attention Implementation | Time per Iteration |
| --- | --- |
| Xformer Attention | 1.9 |
| Flash Attention | 1.75 |
| Default* | 3s |

<sup>*</sup>xformer and flash attention supported batch size of 4 while default attention only supported batch size 2.

### Sequence Length - 2048
| Attention Implementation | Time per Iteration |
| --- | --- |
| Xformer Attention | 4.0s |
| Flash Attention | 3.6s |
| Default* | 9.9s |

<sup>*</sup>xformer and flash attention supported turning gradient checkpointing off while default attention required gradient checkpointing to be on.

### Sequence Length - 4096
| Attention Implementation | Time per Iteration |
| --- | --- |
| Xformer Attention | TODO |
| Flash Attention | TODO |

## Effect of Peft Approaches
We use LoRA to fine-tune the santacoder model. We follow the same setup as above and measure time per iteration.

| Peft | Time per Iteration |
| --- | --- |
| None | 3.6 |
| LoRA (31% trainable parameters) | 3.1 |
