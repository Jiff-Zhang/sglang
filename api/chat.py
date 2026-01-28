try:
    from sglang.srt.logger import logging
except:
    pass

import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from openai import OpenAI, ChatCompletion
from transformers import AutoTokenizer
import tiktoken
import torch.multiprocessing as mp
from concurrent.futures import Executor, ThreadPoolExecutor, as_completed
from functools import partial
from sglang.srt.server_args import ServerArgs
import sglang as sgl
import dataclasses
# from fastchat.model.model_adapter import get_conversation_template

def load_model(server_args: ServerArgs):
    # model = sgl.Engine(**dataclasses.asdict(server_args))
    model = sgl.Engine(server_args=server_args)
    # print('@@@@',model)
    # raise Exception('stop')
    return model

def query_llm(prompts, model_name, model, tokenizer, client=None, temperature=0.1, max_new_tokens=128, stop=None, apply_template=True):
    if apply_template:
        with ThreadPoolExecutor(max_workers=mp.cpu_count()//2) as executor:
            # 提交任务到线程池。
            # TODO:
            if 'DeepSeek-V3.1' in model_name:
                # v3.1
                add_generation_prompt = True
                thinking = True
                # thinking = False
            else:
                # v3 bug exists: to align with baseline which run with add_generation_prompt=False
                add_generation_prompt = False
                thinking = False
            futures = [
                executor.submit(
                    partial(
                        tokenizer.apply_chat_template,
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=add_generation_prompt,
                        thinking=thinking
                    )
                )
                for prompt in prompts
            ]
            prompts = [
                future.result()
                for future in tqdm(futures, total=len(futures), desc="Apply template")
                # for future in tqdm(as_completed(futures), total=len(futures), desc="Apply template")
            ]
        # prompts = [
        #     tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
        #     for prompt in tqdm(prompts, desc="Apply template")
        # ]

    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        # "max_new_tokens": 10,
        "stop": stop,
        # "top_p": args.top_p,
        # "top_k": args.top_k,
        "n": 1,
    }

    outputs = model.generate(prompts, sampling_params)
    responses = [output['text'].strip() for output in outputs]
    return responses

def main(server_args, args):
    tokenizer = AutoTokenizer.from_pretrained(server_args.model_path, trust_remote_code=True)
    model = load_model(server_args)
    prompt = input("User >> ")
    while prompt != "[exit]":
        prompts = [prompt]
        responses = query_llm(
            prompts,
            args.model_name,
            model,
            tokenizer,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            apply_template=args.apply_template
        )
        response = responses[0]
        print(f"Assistant >> {response}")
        prompt = input("User >> ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument("--temperature", "-t", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", "-mt", type=int, default=128)
    parser.add_argument("--apply_template", action='store_true')
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    main(server_args, args)
