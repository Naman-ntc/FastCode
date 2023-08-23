import json
import random

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

from math import ceil

# borrowed and modified from https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/finetuning/APPS/apps_dataset.py


class APPSDataset(torch.utils.data.Dataset):
    def __init__(self, data_args, tokenizer):
        self.data_args = data_args
        split = (
            f"train"
            if data_args.max_total_samples is None
            else f"train[:{data_args.max_total_samples}]"
        )
        self.dataset = load_dataset(
            "codeparrot/apps", split=split, cache_dir=data_args.cache_dir
        )
        self.dataset.shuffle(seed=data_args.seed)

        self.dataset = self.dataset.to_pandas()
        platforms = self.dataset["url"].str.split(".")
        platforms0 = platforms.str[0].str.split("/").str[-1]
        platforms0[platforms0.isin(["open", "www"])] = platforms[
            platforms0.isin(["open", "www"])
        ].str[1]
        self.dataset["platform"] = platforms0
        self.dataset["contains_fn_name"] = self.dataset["input_output"].apply(
            lambda x: "fn_name" in x
        )

        if self.data_args.no_fn_subset:
            self.dataset = self.dataset[
                self.dataset.platform.isin(["codeforces", "codechef", "atcoder"])
            ]
        elif self.data_args.partial_fn_subset:
            raise NotImplementedError("Partial fn subset not implemented yet")

        self.max_tokens = data_args.block_size
        self.tokenizer = tokenizer

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.samples = []

        self._initialize()

    def _initialize(self):
        all_samples = []
        skipped_problems = []

        all_samples_dict = {}  # Mapping from question_fname to list of samples
        count = 0

        for idx in tqdm(range(len(self.dataset))):
            sample = self.dataset.iloc[idx]

            # question
            question_str = sample["question"]

            # solutions
            try:
                solutions = json.loads(sample["solutions"])
            except ValueError:
                skipped_problems.append(idx)
                continue

            # starter code
            starter_code = (
                "" if len(sample["starter_code"]) == 0 else sample["starter_code"]
            )
            try:
                input_outpout = json.loads(sample["input_output"])
                fn_name = (
                    None
                    if not input_outpout.get("fn_name")
                    else input_outpout["fn_name"]
                )
            except ValueError:
                fn_name = None

            answer_type = (
                "\nUse Standard Input format\n"
                if not fn_name
                else "\nUse Call-Based format\n"
            )

            # Read all the solutions
            for solution in solutions:
                # remove samples with long questions
                q_str = (
                    "\nQUESTION:\n"
                    + question_str
                    + "\n"
                    + starter_code
                    + "\n"
                    + answer_type
                    + "\nANSWER:\n"
                )
                q_str_tokenized_inputs = self.tokenizer(q_str)["input_ids"]
                if len(q_str_tokenized_inputs) >= self.max_tokens:
                    count += 1
                    continue

                solution_str_tokenized_inputs = self.tokenizer(solution)[
                    "input_ids"
                ] + [self.tokenizer.eos_token_id]
                sample = [q_str_tokenized_inputs, solution_str_tokenized_inputs]
                all_samples.append(sample)

                if question_str in all_samples_dict:
                    all_samples_dict[question_str].append(sample)
                else:
                    all_samples_dict[question_str] = [sample]

        print(f"Loaded {len(all_samples)} samples")
        print(f"Skipped {len(skipped_problems)} problems because no solution was found")
        print(f"Skipped {count} problems because the prompt was too long")
        self.samples = all_samples
        self.samples_dict = all_samples_dict

    def __len__(self):
        return len(self.samples)

    def _pack_samples(self, idx):
        input_ids = []
        label_ids = []

        curr_num_tokens = 0
        c_q_tokenized, curr_a_tokenized = self.samples[idx]

        while curr_num_tokens < self.max_tokens:
            input_ids.extend(c_q_tokenized)
            label_ids.extend([-100] * len(c_q_tokenized))

            curr_num_tokens += len(c_q_tokenized)

            input_ids.extend(curr_a_tokenized)
            label_ids.extend(curr_a_tokenized)

            curr_num_tokens += len(curr_a_tokenized)

            c_q_tokenized, curr_a_tokenized = random.choice(self.samples)

        assert len(input_ids) == len(label_ids)

        input_ids = input_ids[: self.max_tokens]
        label_ids = label_ids[: self.max_tokens]

        return input_ids, label_ids

    def __getitem__(self, idx):
        input_ids, label_ids = self._pack_samples(idx)
        return {
            "input_ids": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(label_ids),
        }


if __name__ == "__main__":
    import json

    from data.apps.apps_arguments import APPSDataArguments
    from transformers import AutoTokenizer

    # APPSDataArguments.max_total_samples = 10
    setattr(APPSDataArguments, "seed", 0)
    setattr(APPSDataArguments, "cache_dir", None)
    setattr(APPSDataArguments, "no_fn_subset", False)
    setattr(APPSDataArguments, "max_total_samples", 20)

    tokenizer = AutoTokenizer.from_pretrained(
        "bigcode/santacoder",
        use_auth_token=True,
        trust_remote_code=True,
    )

    dataset = APPSDataset(APPSDataArguments, tokenizer)

    for example in range(5):
        example = dataset[example]
        labels = example["labels"]
        labels[labels == -100] = tokenizer.eos_token_id
        decoded_labels = tokenizer.decode(labels)
        decoded_labels = decoded_labels.replace(tokenizer.eos_token, "")
        print(f"labels {'-' * 10}:\n{tokenizer.decode(labels)}")
        print("#" * 50)
