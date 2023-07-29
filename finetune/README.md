# FastCode/FineTune

Performance benchmarks for training code generation models. We compare performance of the model across different training argument combinations and GPUs. For each configuration, we measure the time per iteration with a batch size of 256 over a rack of 8 GPUs.

## V100 (32GB)
| Gradient Checkpointing | Gradient Accumulation Steps | Per Device Train Batch Size | Time per Iteration |
| --- | --- | --- | --- |
| True | 32 | 1 | 63.97s |

## A100 (40GB)
| Gradient Checkpointing | Gradient Accumulation Steps | Per Device Train Batch Size | Time per Iteration |
| --- | --- | --- | --- |
| True | 16 | 2 | 5.2s |
| False | 32 | 1 | 4.9s |
| False | 16 | 2 | 4.1s |

## A6000 (48GB)
| Gradient Checkpointing | Gradient Accumulation Steps | Per Device Train Batch Size | Time per Iteration |
| --- | --- | --- | --- |
| False | 32 | 1 | 20.3s |
| False | 16 | 2 | 19.3s |
| False | 8 | 4 | 18.3s |