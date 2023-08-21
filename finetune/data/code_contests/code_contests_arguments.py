from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CodeContestsArguments:
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
