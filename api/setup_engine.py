try:
    from sglang.srt.logger import logging
except:
    pass

import argparse
from sglang.srt.server_args import ServerArgs
import sglang as sgl
import os
import pickle

os.environ["export GLOO_SOCKET_IFNAME"] = "eth0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["SGL_ENABLE_JIT_DEEPGEMM"] = "false"
os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


def load_model(server_args: ServerArgs):
    print(f"loading model with ServerArgs: {server_args}")
    model = sgl.Engine(server_args=server_args)
    return model

'''
    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "n": 1,
    }

    outputs = model.generate(prompts, sampling_params)
'''

def talk(model):
    prompt = input("User >> ")
    while prompt != "[exit]":
        prompts = [prompt]
        outputs = model.generate(prompts, sampling_params)
        responses = [output['text'].strip() for output in outputs]
        response = responses[0]
        print(f"Assistant >> {response}")
        prompt = input("User >> ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", "-t", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", "-mt", type=int, default=128)
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.model_path = os.path.realpath(args.model_path)

    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "n": 1,
    }

    mf_dir = os.path.join(args.model_path, 'mf')
    os.makedirs(mf_dir, exist_ok=True)
    with open(os.path.join(mf_dir, 'sampling_params.pkl'), 'wb') as f:
        pickle.dump(sampling_params, f)
    with open(os.path.join(mf_dir, 'sampling_params.pkl'), 'rb') as f:
        sampling_params = pickle.load(f)

    server_args = ServerArgs.from_cli_args(args)
    with open(os.path.join(mf_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(server_args, f)
    with open(os.path.join(mf_dir, 'args.pkl'), 'rb') as f:
        server_args = pickle.load(f)

    model = load_model(server_args)
    talk(model, sampling_params)
