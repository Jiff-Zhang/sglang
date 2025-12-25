try:
    from sglang.srt.logger import logging
except:
    pass

import argparse
from sglang.srt.server_args import ServerArgs
import sglang as sgl
import pickle


def load_model(server_args: ServerArgs):
    print(f"loading model with ServerArgs: {server_args}")
    model = sgl.Engine(server_args=server_args)
    return model

'''
    sampling_params = {
        "temperature": 0.6,
        "max_new_tokens": 20480,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0,
        "n": 1,
    }

    outputs = model.generate(prompts, sampling_params)
'''

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # ServerArgs.add_cli_args(parser)
    # args = parser.parse_args()
    # server_args = ServerArgs.from_cli_args(args)
    # with open('/ssd01/workspace/sglang-n/exp/args.pkl', 'wb') as f:
    #     pickle.dump(server_args, f)
    with open('/ssd01/workspace/sglang-n/exp/args.pkl', 'rb') as f:
        server_args = pickle.load(f)
    load_model(server_args)
