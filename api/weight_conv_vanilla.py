import safetensors
from safetensors.torch import save_file
import math
import torch
from torch import nn
from torch.nn import functional as F
import json
import os
import glob, re

from collections import defaultdict
from tqdm import tqdm
from sparseopt.attns.act_sparse_nbits import MFSparseNbits, QuantTool
from sparseopt.compress.compress_model import CompressOPT
from sparseopt.compress.compress_linear import CompressedLinear
from sparseopt.compress.compress_config import CompressConfig, LinearConfig, SparseQuantizeConfig

from einops import rearrange
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM, DeepseekV3DecoderLayer
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from transformers.integrations.finegrained_fp8 import FP8Linear
from datasets import load_dataset, load_from_disk
import typing
from typing import Optional, Union, List, Tuple, Dict, Mapping, Any, Callable
import copy
import shutil
from functools import partial
from collections import OrderedDict
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

def load_weight_map(index_file):
    with open(index_file) as f:
        weight_map = json.load(f)["weight_map"]
    weight_map_ref = defaultdict(list)
    for w, fpath in weight_map.items():
        weight_map_ref[fpath].append(w)
    for fpath, w_list in weight_map_ref.items():
        print(f"{fpath}: {len(w_list)}")
    return weight_map, weight_map_ref

def load_weights(in_weight_dir):
    index_file = os.path.join(in_weight_dir, "model.safetensors.index.json")
    weight_map, weight_map_ref = load_weight_map(index_file)
    
    weights = defaultdict(dict)
    for fpath, w_list in tqdm(weight_map_ref.items(), desc="Loading weights"):
        in_file = os.path.join(in_weight_dir, fpath)
        with safetensors.safe_open(in_file, framework="pt", device="cpu") as f:
            for w_name in w_list:
                weights[fpath][w_name] = f.get_tensor(w_name)
    return weights, weight_map, weight_map_ref

def get_tensor(weights, weight_map, name):
    return weights[weight_map[name]][name]

def set_tensor(weights, weight_map, weight_map_ref, name, tensor, key=None):
    if key is not None:
        weight_map[name] = weight_map[key]
        weight_map_ref[weight_map[name]].append(name)
    weights[weight_map[name]][name] = tensor

class ModelSaver:
    def __init__(self, config, out_dir: str):
        num_layers = config["num_hidden_layers"]
        self.moe_layers = [
            i for i in range(
                # config["first_k_dense_replace"],
                config.get("first_k_dense_replace", 0),
                num_layers,
                # config["moe_layer_freq"]
                config.get("moe_layer_freq", 1)
            )
        ]
        self.normal_layers = [i for i in range(num_layers)]
        self.normal_layers = list(filter(
            lambda i: i not in self.moe_layers, self.normal_layers
        ))
        self.moe_split = 6
        self.total = (
            # embedding + lm_head
            1 +
            # normal layers
            len(self.normal_layers) +
            # moe layers
            self.moe_split * len(self.moe_layers)
        )
        self.num_saved = 0
        self.param_refs = {
            "metadata": {
                "total_size": 0
            },
            "weight_map": {}
        }
        self.out_dir = out_dir

    def save(self, params, layer_idx: int=None):
        os.makedirs(self.out_dir, exist_ok=True)
        if layer_idx is None or layer_idx in self.normal_layers:
            self.num_saved += 1
            filename = f"model-{self.num_saved:06d}-of-{self.total:06d}.safetensors"
            print(f"Saving param to {filename} ...")
            save_file(params, os.path.join(self.out_dir, filename))
            self.param_refs["weight_map"].update({name: filename for name in params.keys()})
            # self.param_refs["metadata"]["total_size"] += os.path.getsize(
            #     os.path.join(self.out_dir, filename)
            # )
            self.param_refs["metadata"]["total_size"] += sum(
                [v.numel() for v in params.values()]
            )
        else:
            num_to_split = math.ceil(len(params) / self.moe_split)
            for i in range(self.moe_split):
                param_to_save = {
                    k: params[k]
                    for k in list(params.keys())[i*num_to_split:(i+1)*num_to_split]
                }
                self.save(param_to_save, layer_idx=None)
                
def run(
    in_weight_dir: str,
    out_weight_dir: str,
    bank_size: int,
    mf_linear_config: Dict[str, float],
    layers: List[str],
    partial_save: bool=False
):
    mf_tools = {
        key: MFSparseNbits(
            sparsity=config["weight"]["sparsity"],
            bank_size=bank_size,
            sparse_mode="per_bank",
            quant_mode="per_bank",
            dtypes={
                "high": "none" if config["weight"]["high_bits"] == 16 else f"int{config['weight']['high_bits']}",
                "low": "zero" if config["weight"]["low_bits"] == 0 else f"int{config['weight']['low_bits']}"
            },
            quant_symmetric=True,
            quant_masked=True,
            hardware=True
        )
        for key, config in mf_linear_config.items()
    }
    weights, weight_map, weight_map_ref = load_weights(in_weight_dir)

    if partial_save:
        with open(os.path.join(in_weight_dir, "config.json"), "r") as fid:
            config = json.load(fid)
        weight_map_dict = {
            i: {
                name: value for name, value in weight_map.items()
                if f"model.layers.{i}." in name
            } for i in range(0, config["num_hidden_layers"])
        }
        weight_map_dict[None] = {
            name: value for name, value in weight_map.items()
            if f"model.layers." not in name
        }
        saver = ModelSaver(config, out_weight_dir)
    else:
        weight_map_dict = {None: weight_map}
        saver = None
    ori_weight_map = weight_map

    with open(os.path.join(in_weight_dir, "config.json"), "r") as fid:
        model_config = json.load(fid)
        if "quantization_config" in model_config:
            w_block = model_config["quantization_config"]["weight_block_size"]
        else:
            w_block = None
    for layer_idx, weight_map in weight_map_dict.items():
        for name in tqdm(list(weight_map.keys())[:], desc="Processing weights"):
            if not name.endswith("weight"):
                continue
            if any(l in name for l in layers):
                w_name = name
                s_name = name.replace(".weight", ".weight_scale_inv")
                weight = get_tensor(weights, weight_map, w_name)
                has_scale = s_name in weight_map
                if has_scale:
                    assert weight.dtype not in [torch.bfloat16, torch.float16, torch.float32, torch.float]
                    scale = get_tensor(weights, weight_map, s_name)
                    scale = scale.repeat_interleave(w_block[0], dim=0)
                    scale = scale.repeat_interleave(w_block[1], dim=1)
                    weight = weight.to(scale.dtype) * \
                        scale[:weight.size(0), :weight.size(1)]
                else:
                    assert weight.dtype in [torch.bfloat16, torch.float16, torch.float32, torch.float]
                tqdm.write(f"### {w_name} with shape {list(weight.shape)} ###")
                    
            # if name.endswith(".weight_scale_inv"):
            #     s_name = name
            #     w_name = name.replace(".weight_scale_inv", ".weight")
                
            #     scale = get_tensor(weights, weight_map, s_name)
            #     weight = get_tensor(weights, weight_map, w_name)
                
            #     scale = scale.repeat_interleave(w_block[0], dim=0)
            #     scale = scale.repeat_interleave(w_block[1], dim=1)
            #     weight = weight.to(scale.dtype) * \
            #         scale[:weight.size(0), :weight.size(1)]
                ori_weight = weight

                mf_tool = mf_tools["__default__"]
                for k in mf_tools:
                    if re.search(k, w_name):
                        mf_tool = mf_tools[k]
                        break
                
                # quantize: int8 weight + bfloat16 scale
                scale_dtype = torch.float32
                scale_dtype = torch.bfloat16
                
                # split to high and low
                hweight = mf_tool.sparse_tool.transform.preprocess(weight)
                hweight, lweight, mask = mf_tool.sparse_tool(hweight)
                hweight = mf_tool.sparse_tool.transform.postprocess(hweight)
                lweight = mf_tool.sparse_tool.transform.postprocess(lweight)
                mask = mf_tool.sparse_tool.transform.postprocess(mask)
                # quant high
                hweight = mf_tool.high_quant_tool.transform.preprocess(hweight)
                hweight, hscale = mf_tool.high_quant_tool.sym_quant(hweight)
                hweight = mf_tool.high_quant_tool.transform.postprocess(hweight).to(torch.int8)
                hscale = mf_tool.high_quant_tool.transform.postprocess(hscale).to(scale_dtype)
                # quant low
                if mf_tool.sparsity > 0 and mf_tool.dtypes["low"] != "zero":
                    lweight = mf_tool.low_quant_tool.transform.preprocess(lweight)
                    lweight, lscale = mf_tool.low_quant_tool.sym_quant(lweight)
                    lweight = mf_tool.low_quant_tool.transform.postprocess(lweight).to(torch.int8)
                    lscale = mf_tool.low_quant_tool.transform.postprocess(lscale).to(scale_dtype)
                    if mask_in_id:
                        # [O, NB, BankSize]
                        mask = mf_tool.sparse_tool.transform.preprocess(mask)
                        idx, idy, idz = torch.where(mask == 1)
                        mask_id = idz.view(mask.size(0), mask.size(1), -1)
                        mask_n = torch.zeros(
                            (mask_id.size(0), mask_id.size(1), mf_tool.sparse_tool.bank_size),
                            dtype=torch.int8
                        )
                        mask_n.scatter_(2, mask_id, 1)
                        mask_n = mf_tool.sparse_tool.transform.postprocess(mask_n)
                        mask = mf_tool.sparse_tool.transform.postprocess(mask)
                        assert (mask == mask_n).all()
                else:
                    # module.lweight = torch.tensor(0, dtype=torch.int8)
                    lscale = torch.tensor(0, dtype=scale_dtype)
                    mask = torch.tensor(1, dtype=torch.int8)

                weight = (hweight * mask + lweight * (1 - mask)).to(torch.int8)
                
                new_weight = weight.to(scale_dtype) * \
                    hscale.to(scale_dtype).repeat_interleave(
                        mf_tool.bank_size, dim=-1
                    ) * mask.to(scale_dtype)

                if mf_tool.sparsity > 0 and mf_tool.dtypes["low"] != "zero":
                    new_weight += weight.to(scale_dtype) * \
                        lscale.to(scale_dtype).repeat_interleave(
                            mf_tool.bank_size, dim=-1
                        ) * (1 - mask).to(scale_dtype)
                        
                ori_weight = ori_weight.view(-1)
                new_weight = new_weight.view(-1)
                # diff = ((new_weight - ori_weight).abs() / ori_weight.abs())
                diff = (new_weight - ori_weight).abs()
                # diff[ori_weight == 0] = 0
                # value, index = diff.max(dim=0)
                # ori_value = ori_weight[index]
                # new_value = new_weight[index]
                tqdm.write(
                    # f"{w_name}: {value.item(), ori_value.item(), new_value.item()}"
                    f"\tdiff mean: {diff.mean().item():.5f}\tdiff max: {diff.max().item():.5f}"
                )

                # update weights
                set_tensor(
                    weights,
                    weight_map,
                    weight_map_ref,
                    s_name,
                    hscale.to(scale_dtype),
                    key=s_name if has_scale else w_name
                )
                set_tensor(
                    weights,
                    weight_map,
                    weight_map_ref,
                    w_name,
                    weight.to(torch.int8)
                )
                if mf_tool.sparsity > 0 and mf_tool.dtypes["low"] != "zero":
                    set_tensor(
                        weights,
                        weight_map, 
                        weight_map_ref,
                        s_name.replace(".weight_scale_inv", ".weight_lscale_inv"),
                        lscale.to(scale_dtype),
                        key=s_name
                    )
                    if mask_in_id:
                        set_tensor(
                            weights,
                            weight_map, 
                            weight_map_ref,
                            w_name.replace(".weight", ".mask_id"),
                            mask_id.to(torch.int8),
                            key=w_name
                        )
                    else:
                        set_tensor(
                            weights,
                            weight_map, 
                            weight_map_ref,
                            w_name.replace(".weight", ".mask"),
                            mask.to(torch.int8),
                            key=w_name
                        )
        if partial_save:
            saver.save(
                {
                    name: get_tensor(weights, weight_map, name)
                    for name in weight_map.keys()
                },
                layer_idx=layer_idx
            )
            for name in weight_map.keys():
                set_tensor(weights, weight_map, weight_map_ref, name, None)

    if partial_save:
        weight_map = saver.param_refs["weight_map"]
    else:
        weight_map = ori_weight_map
        save_weights(weights, out_weight_dir)
    
    save_extra(
        in_weight_dir,
        out_weight_dir,
        weight_map,
        mf_linear_config,
        bank_size,
    )

def save_weights(weights, out_weight_dir):
    os.makedirs(out_weight_dir, exist_ok=True)
    for fpath, w_dict in tqdm(weights.items(), desc="Saving weights"):
        out_file = os.path.join(out_weight_dir, fpath)
        safetensors.torch.save_file(weights[fpath], out_file)
        tqdm.write(f"Saved {len(w_dict)} weights to {out_file}")

def save_extra(
    in_weight_dir, out_weight_dir, weight_map, 
    mf_linear_config, bank_size
):
    file_list = [
        "configuration.json",
        "configuration_deepseek.py",
        "configuration_minimax_m2.py",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "modeling_deepseek.py",
        "modeling_minimax_m2.py",
        "inference",
        "config.json",
        "model.safetensors.index.json",
    ]
    smooth = False
    for filepath in file_list:
        if os.path.exists(os.path.join(in_weight_dir, filepath)):
            if filepath == "config.json":
                with open(os.path.join(out_weight_dir, filepath), mode="w") as ofid, \
                        open(os.path.join(in_weight_dir, filepath), mode="r") as ifid:
                    info = json.load(ifid)
                    info["quantization_config"] = {
                        "activation_scheme": "dynamic",
                        "quant_method": "blockwise_int8",
                        "mf_linear_config": mf_linear_config,
                        "weight_block_size": [
                            1,
                            bank_size
                        ]
                    }
                    json.dump(info, ofid, indent=2, ensure_ascii=True)
            elif filepath == "model.safetensors.index.json":
                with open(os.path.join(out_weight_dir, filepath), mode="w") as ofid, \
                        open(os.path.join(in_weight_dir, filepath), mode="r") as ifid:
                    info = json.load(ifid)
                    info["weight_map"] = weight_map
                    json.dump(info, ofid, indent=2, ensure_ascii=True)
            elif os.path.isdir(os.path.join(in_weight_dir, filepath)):
                shutil.copytree(
                    os.path.join(in_weight_dir, filepath),
                    os.path.join(out_weight_dir, filepath),
                    dirs_exist_ok=True
                )
            else:
                shutil.copy(
                    os.path.join(in_weight_dir, filepath),
                    os.path.join(out_weight_dir, filepath)
                )

if __name__ == "__main__":
    in_weight_dir = "/ssd01/models/DeepSeek-V3.1-Terminus"
    out_weight_dir = "/ssd01/models/DeepSeek-V3.1-Terminus-MF-Int8"
    out_weight_dir = "/ssd01/models/DeepSeek-V3.1-Terminus-MF-W8xH8L3"
    # out_weight_dir = "/ssd01/models/DeepSeek-V3.1-Terminus-MF-Linear_WInt8-MOE_W8xH8L3"
    
    in_weight_dir = "/ssd01/models/DeepSeek-V3.2-Exp"
    # out_weight_dir = "/ssd01/models/DeepSeek-V3.2-Exp-MF-Int8"
    # out_weight_dir = "/ssd01/models/DeepSeek-V3.2-Exp-MF-W8xH8L3"
    out_weight_dir = "/ssd01/models/DeepSeek-V3.2-Exp-MF-Linear_WInt8-MOE_W8xH8L3"

    in_weight_dir = "/ssd01/models/Qwen3-235B-A22B-Instruct-2507"
    out_weight_dir = "/ssd01/models/Qwen3-235B-A22B-Instruct-2507-MF-Int8"
    out_weight_dir = "/ssd01/models/Qwen3-235B-A22B-Instruct-2507-MF-Linear_WInt8-MOE_W8xH8L3"

    in_weight_dir = "/ssd01/models/MiniMax-M2.1"
    out_weight_dir = "/ssd01/models/MiniMax-M2.1-MF-Int8"
    # out_weight_dir = "/ssd01/models/MiniMax-M2.1-MF-Linear_WInt8-MOE_W8xH8L3"
    
    # vanilla version
    bank_size = 64
    high_bits = 8
    low_bits = 3
    mask_in_id = False
    mask_in_id = True
    partial_save = False
    partial_save = True
    smooth = False
    mf_linear_config = {
        "__default__": {
            "weight": {
                "sparsity": 0,
                "high_bits": high_bits,
                "low_bits": low_bits,
                "mask_in_id": mask_in_id
            },
            "input": {
                "sparsity": 0,
                "high_bits": high_bits,
                "low_bits": 0,
                "mf_format": True
            },
            "smooth": smooth
        },
        # ".*experts": {
        #     "weight": {
        #         "sparsity": 0.875,
        #         "high_bits": high_bits,
        #         "low_bits": low_bits,
        #         "mask_in_id": True
        #     },
        #     "input": {
        #         "sparsity": 0,
        #         "high_bits": high_bits,
        #         "low_bits": 0,
        #         "mf_format": True
        #     },
        #     "smooth": smooth
        # }
    }
    layers = [
        "gate_proj", "up_proj", "down_proj",
        "k_proj", "v_proj", "q_proj", "o_proj",
        "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj",
        "w1", "w2", "w3", # MiniMax-M2.1
    ]
    
    run(
        in_weight_dir,
        out_weight_dir,
        bank_size=bank_size,
        mf_linear_config=mf_linear_config,
        layers=layers,
        partial_save=partial_save
    )