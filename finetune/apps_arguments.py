from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )

    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    use_flash_attn: bool = field(
        default=False,
        metadata={
            "help": (
                "Use flash attention implementation."
                "This is set_true argument, do not change default"
            )
        },
    )

    use_xformer_attn: bool = field(
        default=False,
        metadata={
            "help": (
                "Use xformer memory efficient attention implementation."
                "This is set_true argument, do not change default"
            )
        },
    )

@dataclass
class APPSDataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    max_total_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of total examples to this "
                "value if set."
            )
        },
    )

    block_size: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )

    eval_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )

    no_fn_subset: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Only use the subset of training data from codeforces, codechef, and atcoder platforms"
        },
    )

    partial_fn_subset: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Only use the subset of training data from codeforces, codechef, and atcoder platforms"
        },
    )
    

@dataclass
class ModelSpecificArguments:
    """
    Arguments pertaining to the specific model to be used.
    """

    scale_attention_softmax_in_fp32 : Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Scale the attention softmax in fp32. This argument is only relevant for GPTBigCodeForCausalLM."
            )
        },
    )

    attention_softmax_in_fp32 : Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Compute the attention softmax in fp32. This argument is only relevant for GPTBigCodeForCausalLM."
            )
        },
    )

    alibi : Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Use alibi positional embeddings. This argument is only relevant for MPTForCausalLM."
            )
        },
    )

    attn_impl : Optional[str] = field(
        default="triton",
        metadata={
            "help": (
                "Attention implementation. This argument is only relevant for MPTForCausalLM."
            )
        },
    )