import os
import sys
import json
import random
import fnmatch
import warnings
import subprocess

import torch
import datasets
import numpy as np
import transformers
from transformers import HfArgumentParser

from combine_generations import main as combine_generations
from generation_arguments import EvalArguments

sys.path.append("../evaluation/bigcode-evaluation-harness")
from lm_eval import tasks
from lm_eval.tasks import ALL_TASKS


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def pattern_match(patterns, source_list):
    """Returns a list containing all values of the source_list that
    match at least one of the patterns"""
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def parse_args():
    parser = HfArgumentParser(EvalArguments)

    parser.add_argument(
        "--model",
        default="codeparrot/codeparrot-small",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Model revision to use",
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Use the token generated when running `huggingface-cli login` (necessary for private model).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Use a model with custom code, this requires executing code by the author of the model.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        choices=MultiChoice(ALL_TASKS),
        help=f"Evaluation tasks from {ALL_TASKS}",
    )
    parser.add_argument(
        "--instruction_tokens",
        default=None,
        help="A series of instruction tokens used for instruction-tuning benchamrks separated by comma e.g. <user_message>,<end_user_message>,<assistant_message>",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation on each worker, can be larger for HumanEval",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=1024,
        help="Maximum length of generated sequence (prompt+generation)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        help="Model precision, from: fp32, fp16 or bf16",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset before evaluation (useful for distributed inference)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only solve the first limit samples in the benchmark (useful with randomize dataset)",
    )
    parser.add_argument(
        "--postprocess",
        action="store_false",
        help="Postprocess model outputs before execution, always on except during generation tests",
    )
    parser.add_argument(
        "--base_directory",
        type=str,
        default="model_outputs",
        help="Base directory for saving the model outputs",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="generations.json",
        help="Path for saving the code generations",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to distribute inference on",
    )
    parser.add_argument(
        "--allow_code_execution",
        action="store_true",
        help="Allow code evaluation to execute external/untrusted Python code on your machine",
    )
    args = parser.parse_args()

    precision_map = {
        "fp32": "float32",
        "fp16": "float16",
        "bf16": "bfloat16",
    }

    args.precision = precision_map[args.precision]
    args.tasks = pattern_match(args.tasks.split(","), ALL_TASKS)
    assert (
        len(args.tasks) == 1
    ), f"Only one task is supported at the moment, you gave {args.tasks}"
    args.task_name = args.tasks[0]

    assert args.instruction_tokens is None, "Instruction tokens are not supported yet"
    return args


def evaluate_generations(task, args, generations, references):
    dataset = task.get_dataset()
    if len(generations[0]) > args.n_samples:
        generations = [l[: args.n_samples] for l in generations]
        warnings.warn(
            f"Number of tasks wasn't proportional to number of devices, we removed extra predictions to only keep nsamples={self.args.n_samples}"
        )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.allow_code_execution and task.requires_execution:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    print("Evaluating generations...")
    evaluation_results = task.process_results(generations, references)
    return evaluation_results


def get_references(task, args):
    dataset = task.get_dataset()
    n_tasks = args.limit if args.limit else len(dataset)
    references = [task.get_reference(dataset[i]) for i in range(n_tasks)]
    return references


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    args = parse_args()
    task = tasks.get_task(args.task_name)

    if task.requires_execution and not args.allow_code_execution:
        from lm_eval.evaluator import _WARNING

        raise ValueError(_WARNING)

    references = get_references(task, args)

    if args.limit is None:
        print(
            f"limit not set -- using full dataset with size {len(task.get_dataset())}"
        )
        data_size = len(task.get_dataset())
    else:
        data_size = args.limit

    all_arguments = sys.argv[1:]
    processes = []
    generations_paths = []

    ensure_dir(args.base_directory)
    for gpu_idx in range(args.num_gpus):
        start = int(data_size * (gpu_idx / args.num_gpus))
        end = int(data_size * ((gpu_idx + 1) / args.num_gpus))
        generations_path = (
            f"{args.base_directory}/generations_{args.exp_name}_{gpu_idx}.json"
        )
        cuda_command = f"export CUDA_VISIBLE_DEVICES={gpu_idx}"
        print_info_command = (
            f"echo 'Running on GPU {gpu_idx} with indices {start} to {end}'"
        )
        run_command = []
        run_command.append("python")
        run_command.append("main_helper.py")
        run_command.extend(all_arguments)
        run_command.extend(["--start", str(start)])
        run_command.extend(["--end", str(end)])
        run_command.extend(["--save_generations_path", generations_path])
        run_command = " ".join(run_command)

        p = subprocess.Popen(
            "; ".join([cuda_command, print_info_command, run_command]),
            shell=True,
        )

        processes.append(p)
        generations_paths.append(generations_path)

    print(f"Started {len(processes)} processes with PIDs {[p.pid for p in processes]}")
    for p in processes:
        p.wait()

    combined_json = f"{args.base_directory}/generations_{args.exp_name}.json"

    combined_generations = combine_generations(generations_paths, combined_json)

    evaluation_results = evaluate_generations(
        task, args, combined_generations, references
    )

    if isinstance(evaluation_results, tuple):
        results, all_results = evaluation_results
    else:
        results = evaluation_results
        all_results = None

    argparse_dict = vars(args)
    with open(f"{args.base_directory}/args_{args.exp_name}.json", "w") as fp:
        json.dump(argparse_dict, indent=4, fp=fp)

    with open(
        f"{args.base_directory}/evaluation_results_{args.exp_name}.json", "w"
    ) as fp:
        json.dump(results, indent=4, fp=fp)

    if all_results is not None:
        with open(
            f"{args.base_directory}/all_evaluation_results_{args.exp_name}.json", "w"
        ) as fp:
            json.dump(all_results, indent=4, fp=fp)

    return


if __name__ == "__main__":
    main()
