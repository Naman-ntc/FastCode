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

from combine_generations import main as combine_generations, format_solution
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
        default="exp",
        help="Path for saving the code generations",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to distribute inference on",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of gpus to shard the model across. Should divide num_gpus evenly",
    )
    parser.add_argument(
        "--allow_code_execution",
        action="store_true",
        help="Allow code evaluation to execute external/untrusted Python code on your machine",
    )
    parser.add_argument(
        "--eval_mode_only",
        action="store_true",
        help="Only run evaluation, skip generation",
    )
    parser.add_argument(
        "--load_gen_file",
        type=str,
        default=None,
        help="generations file for doing evaluation",
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

    if args.tensor_parallel_size > 1:
        assert (
            args.num_gpus % args.tensor_parallel_size == 0
        ), "num_gpus must be divisible by tensor_parallel_size"

    assert args.instruction_tokens is None, "Instruction tokens are not supported yet"
    return args


def evaluate_generations(task, args, generations, references):
    if len(generations[0]) > args.n_samples:
        generations = [l[: args.n_samples] for l in generations]
        warnings.warn(
            f"Number of tasks wasn't proportional to number of devices, we removed extra predictions to only keep nsamples={args.n_samples}"
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


def load_generations(generations_path):
    with open(generations_path) as fp:
        generations = json.load(fp)
        print(f"generations loaded, {len(generations)} tasks")
    formatted_combined = [[format_solution(s) for s in sols] for sols in generations]
    with open(generations_path.replace(".json", "_formatted.json"), "w") as fp:
        json.dump(formatted_combined, indent=4, fp=fp)
    return generations, formatted_combined


def run_generations(args, data_size):
    all_arguments = sys.argv[1:]
    processes = []
    generations_paths = []

    num_divisions = args.num_gpus // args.tensor_parallel_size

    for iter_idx in range(num_divisions):
        start = int(data_size * (iter_idx / num_divisions))
        end = int(data_size * ((iter_idx + 1) / num_divisions))
        generations_path = (
            f"{args.base_directory}/generations_{args.exp_name}_{iter_idx}.json"
        )
        gpu_list = ",".join(
            [str(i) for i in range(iter_idx, args.num_gpus, num_divisions)]
        )
        cuda_command = f"export CUDA_VISIBLE_DEVICES={gpu_list}"
        print_info_command = (
            f"echo 'Running on GPU {gpu_list} with indices {start} to {end}'"
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

    generations, formatted_generations = combine_generations(
        generations_paths, combined_json
    )
    return generations, formatted_generations


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

    ensure_dir(args.base_directory)

    if args.eval_mode_only:
        print("Skipping generation")
        if args.load_gen_file is None:
            generations_path = f"{args.base_directory}/generations_{args.exp_name}.json"
        else:
            generations_path = args.load_gen_file
        generations, formatted_generations = load_generations(generations_path)
    else:
        generations, formatted_generations = run_generations(args, data_size)

    evaluation_results = evaluate_generations(
        task, args, formatted_generations, references
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
