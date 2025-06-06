import argparse
import ast
import json
import os
import re
import time

import numpy as np

from sglang.api import set_default_backend
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import download_and_cache_file, dump_state_text, read_jsonl
from sglang.srt.server_args import ServerArgs
import sglang as sgl
import dataclasses

INVALID = -9999999


def load_model(server_args: ServerArgs):
    model = sgl.Engine(**dataclasses.asdict(server_args))
    # print('@@@@',model)
    # raise Exception('stop')
    return model

def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def main(server_args: ServerArgs, args):
    # Select backend
    # set_default_backend(select_sglang_backend(args))

    # Read data
    data_path = args.data_path
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    if not os.path.isfile(data_path):
        data_path = download_and_cache_file(url)
    lines = list(read_jsonl(data_path))

    # Construct prompts
    num_questions = args.num_questions
    num_shots = args.num_shots
    few_shot_examples = get_few_shot_examples(lines, num_shots)

    questions = []
    labels = []
    for i in range(len(lines[:num_questions])):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)
    arguments = [{"question": q} for q in questions]

    """
    #####################################
    ######### SGL Program Begin #########
    #####################################

    import sglang as sgl

    @sgl.function
    def few_shot_gsm8k(s, question):
        s += few_shot_examples + question
        s += sgl.gen(
            "answer", max_tokens=512, stop=["Question", "Assistant:", "<|separator|>"]
        )

    #####################################
    ########## SGL Program End ##########
    #####################################
    """
    model = load_model(server_args)

    # Run requests
    tic = time.time()
    sampling_params = {
        "temperature": 0,
        "max_new_tokens": 512,
        "n": 1,
        "stop": ["Question", "Assistant:", "<|separator|>"]
    }
    prompts = [
        few_shot_examples + argument['question']
        for argument in arguments
    ]
    outputs = model.generate(prompts, sampling_params)
    preds = [get_answer_value(output['text'].strip()) for output in outputs]
    """
    states = few_shot_gsm8k.run_batch(
        arguments,
        temperature=0,
        num_threads=args.parallel,
        progress_bar=True,
    )

    preds = []
    for i in range(len(states)):
        preds.append(get_answer_value(states[i]["answer"]))
    """
    latency = time.time() - tic

    # print(f"{preds=}")
    # print(f"{labels=}")

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    # Compute speed
    """
    num_output_tokens = sum(
        s.get_meta_info("answer")["completion_tokens"] for s in states
    )
    output_throughput = num_output_tokens / latency
    """

    # Print results
    print(f"Accuracy: {acc:.3f}")
    print(f"Invalid: {invalid:.3f}")
    print(f"Latency: {latency:.3f} s")
    # print(f"Output throughput: {output_throughput:.3f} token/s")

    # Dump results
    # dump_state_text(f"tmp_output_{args.backend}.txt", states)

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'result.txt'), "a") as fout:
        value = {
            "task": "gsm8k",
            # "backend": args.backend,
            "num_gpus": 1,
            "latency": round(latency, 3),
            "accuracy": round(acc, 3),
            "num_requests": args.num_questions,
            "other": {
                "num_questions": args.num_questions,
                # "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--save-dir", type=str, default="results")
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    main(server_args, args)
    # args = add_common_sglang_args_and_parse(parser)
    # main(args)
