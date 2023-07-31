import math
import warnings
from collections import defaultdict

import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm

class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset
    Multiple copies of the same prompt are sent sequentially. See compute_code for more details.
    The prompt can either be:
    - one prompt: normal code completion
    - two prompts: for infilling mode (prefix, suffix) or instructin-tuning mode (instruction, context)
    """

    def __init__(
        self,
        task,
        dataset,
        tokenizer,
        max_length,
        n_tasks=None,
        n_copies=1,
        prefix="",
    ):
        self.task = task
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_tasks = n_tasks
        self.n_copies = n_copies
        self.prefix = prefix


    def __iter__(self):
        prompts = []
        row_idxs = []
        for sample in range(self.n_tasks):
            dataset_sample = self.dataset[sample]
            prompt_contents = self.task.get_prompt(dataset_sample)
            assert isinstance(prompt_contents, str)
            prompt = self.prefix + prompt_contents
            prompts.append(prompt)
            row_idxs.append(dataset_sample["row_index"])

        return_token_type_ids = None  # default

        outputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            return_token_type_ids=return_token_type_ids,
        )


        for sample in range(self.n_tasks):
            for _ in range(self.n_copies):
                yield {
                    "row_index": row_idxs[sample],
                    "prompt": prompts[sample],
                    "ids": outputs.input_ids[sample],
                    "input_len": outputs.attention_mask[sample].sum(),
                }


def complete_code(
    task,
    model,
    sampling_params,
    dataloader,
    batch_size,
    n_tasks,
    prefix="",
    postprocess=True,
):
    code_gens = defaultdict(list)
    total = math.ceil(n_tasks * dataloader.dataset.n_copies)
    for step, batch in tqdm(enumerate(dataloader), total=total):
        inputs = batch["ids"][:, : batch["input_len"]].tolist()
        num_tokens = len(inputs[0])
        if sampling_params.max_tokens - num_tokens < 0:
            code_gens[int(batch["row_index"][0])].extend([""] * batch_size)
            warnings.warn(
                f"Skipping task {batch['row_index'][0]} because it is too long"
            )
            continue
        sampling_params.max_tokens = sampling_params.max_tokens - num_tokens
        outputs = model.generate(prompt_token_ids=inputs, sampling_params=sampling_params, use_tqdm=False)

        generated_tasks = batch["row_index"].repeat(batch_size)
        generated_texts = [o.text for o in outputs[0].outputs]
        combined_texts = [
            batch["prompt"][0] + generated_text for generated_text in generated_texts
        ]

        for task, text in zip(generated_tasks, combined_texts):
            if postprocess:
                text = task.postprocess_generation(text, int(task.item()))
            code_gens[int(task.item())].append(text)
    
    return code_gens