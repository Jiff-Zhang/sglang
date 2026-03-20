import argparse
from tqdm import tqdm
import safetensors
from glob import glob, glob1
import os

def check(sf_file):
    with safetensors.safe_open(sf_file, framework="pt") as f:
        for k in tqdm(f.keys(), desc=f"Checking {os.path.basename(sf_file)}"):
            v = f.get_tensor(k)
            if v.isnan().any():
                tqdm.write(f"{k} has nan")
            if v.isinf().any():
                tqdm.write(f"{k} has inf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", type=str, required=True)
    args = parser.parse_args()

    for path in glob(os.path.join(args.model_path, "*.safetensors")):
        check(path)