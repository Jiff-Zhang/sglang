import safetensors
from safetensors.torch import save_file
import torch
from torch import nn
from torch.nn import functional as F
import json
import os
import math
import glob

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
from functools import partial, reduce
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
import shutil
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# reuse data
def load_cache(cache_file):
    cache_data = torch.load(cache_file, weights_only=False, map_location='cpu')
    args_cache = cache_data['args']
    kwargs_cache = cache_data['kwargs']
    assert len(args_cache) == len(kwargs_cache)
    return args_cache, kwargs_cache

def disable_torch_init():
    """Disable initialization of Pytorch."""

    def skip(*args, **kwargs):
        pass

    torch.nn.init.normal_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.kaiming_uniform_ = skip

    DeepseekV3ForCausalLM._init_weights = skip

disable_torch_init()

def get_weight(linear: Union[nn.Module, FP8Linear]):
    if isinstance(linear, FP8Linear):
        w_block = [128, 128]
        weight = linear.weight
        scale = linear.weight_scale_inv
        scale = scale.repeat_interleave(w_block[0], dim=0)
        scale = scale.repeat_interleave(w_block[1], dim=1)
        weight = weight.to(scale.dtype) * \
            scale[:weight.size(0), :weight.size(1)]
        return weight
    return linear.weight.clone()

def get_weight_pair(linears: List[Union[nn.Module, FP8Linear]]):
    return torch.cat([get_weight(linear) for linear in linears], dim=0)

def get_compress_config(sparsity, high_bits, low_bits):
    if sparsity == 0:
        low_bits = 0
    compress_config = CompressConfig(
        general_linear=LinearConfig(
            inputs=SparseQuantizeConfig(
                sparsity=0,
                quantize=True,
                high_bits=8,
                low_bits=0,
                bank_size=64,
                group_size=64,
                quant_method="per_bank",
                kv_prefill_compress=False,
                quant_symmetric=True,
                fp8=False
            ),
            weights=SparseQuantizeConfig(
                sparsity=sparsity,
                quantize=True,
                high_bits=high_bits,
                low_bits=low_bits,
                bank_size=64,
                group_size=64,
                quant_method="per_bank",
                kv_prefill_compress=False,
                quant_symmetric=True,
                fp8=False
            ),
            outputs=SparseQuantizeConfig(
                sparsity=0,
                quantize=False,
                high_bits=8,
                low_bits=0,
                bank_size=64,
                group_size=64,
                quant_method="per_bank",
                kv_prefill_compress=False,
                quant_symmetric=True,
                fp8=False
            ),
            smooth=True,
            num_grids=40
        )
    )
    return compress_config

def get_compress_tool(
    linears: List[Union[nn.Module, FP8Linear]], 
    compress_config: CompressConfig
):
    weight = get_weight_pair(linears)
    compressed_linear = CompressedLinear(
        in_features=weight.size(1),
        out_features=weight.size(0),
        bias=False,
        compress_cfg=compress_config.general_linear
    )
    compressed_linear.weight.data.copy_(weight)
    compressopt = CompressOPT(
        module=compressed_linear,
        compress_cfg=compress_config.general_linear
    )
    return compressopt

def add_batch(
    module: nn.Module,
    args: Tuple[torch.Tensor, ...],
    kwargs: Dict[str, typing.Any],
    tool: CompressOPT
):
    tool.add_batch(inp=args[0].detach().cpu(), out=None)

def register_compress_tool(
    layer: DeepseekV3DecoderLayer,
    compress_configs: Dict[str, CompressConfig]
):
    hooks, tools = dict(), dict()
    share_pairs = [
        ("q_a_proj", "kv_a_proj_with_mqa"),
        ("gate_proj", "up_proj")
    ]
    for name, module in layer.named_modules():
        if isinstance(module, (nn.Linear, FP8Linear)):
            # share smooth scale
            pair = []
            for pair_item in share_pairs:
                if any([item in name for item in pair_item]):
                    pair = pair_item
            if len(pair) == 0:
                modules = [module]
                module_names = [name]
            else:
                match_key = list(filter(lambda x: x in name, pair))[0]
                module_names = [name.replace(match_key, item) for item in pair]
                modules = [
                    layer.get_submodule(module_name)
                    for module_name in module_names
                ]
                # print(index, match_key, pair, tools[name].weight_slice)

            if name not in tools:
                compress_config = compress_configs["linear"]
                for key in compress_configs:
                    if key in name:
                        compress_config = compress_configs[key]
                        break
                    
                tools[name] = get_compress_tool(
                    modules, compress_config=compress_config
                )
                wo_list = [0] + [module.weight.size(0) for module in modules]
                setattr(
                    tools[name], "weight_slices",
                    torch.cumsum(torch.tensor(wo_list), dim=0)
                )
                setattr(tools[name], "ori_modules", modules)
                setattr(tools[name], "ori_module_names", module_names)
                for module_name in module_names:
                    tools[module_name] = tools[name]
            
            hooks[name] = module.register_forward_pre_hook(
                partial(add_batch, tool=tools[name]),
                with_kwargs=True
            )
    return hooks, tools

def solve_weight(
    name,
    tool: CompressOPT,
    smooth=True,
    smooth_filters: list = [],
    mask_in_id=False,
):
    if smooth and not any([item in name for item in smooth_filters]):
        if not getattr(tool, "already_smooth", False):
            tool.smooth()
            setattr(tool, "already_smooth", True)
        else:
            print("already smooth !!!")
    
    weight = tool.module.weight
    module_id = getattr(tool, "ori_module_names").index(name)
    weight_slices = getattr(tool, "weight_slices")
    weight = weight[weight_slices[module_id]: weight_slices[module_id + 1]]

    config = tool.compress_cfg.weights
    dtypes = {
        "high": "none" if config.high_bits == 16 else f"int{config.high_bits}",
        "low": "zero" if config.low_bits == 0 else f"int{config.low_bits}"
    }
    mf_tool = MFSparseNbits(
        sparsity = config.sparsity,
        bank_size=config.bank_size,
        sparse_mode="per_bank",
        quant_mode="per_bank",
        dtypes=dtypes,
        quant_symmetric=True,
        quant_masked=True,
        hardware=True
    )

    module = tool.ori_modules[module_id]
    ori_weight = get_weight(module)
    
    assert module.weight.size() == weight.size(), \
        f"{module.weight.size()} != {weight.size()}"
    if not isinstance(module, (nn.Linear, FP8Linear)):
        raise NotImplementedError

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
    module.weight_scale_inv = nn.Parameter(
        hscale.to(module.weight_scale_inv.device, scale_dtype)
    )
    # quant low
    if mf_tool.sparsity > 0 and mf_tool.dtypes["low"] != "zero":
        lweight = mf_tool.low_quant_tool.transform.preprocess(lweight)
        lweight, lscale = mf_tool.low_quant_tool.sym_quant(lweight)
        lweight = mf_tool.low_quant_tool.transform.postprocess(lweight).to(torch.int8)
        lscale = mf_tool.low_quant_tool.transform.postprocess(lscale).to(scale_dtype)
    else:
        lweight, lscale = 0, 0

    weight = (hweight * mask + lweight * (1 - mask)).to(torch.int8)
    module.weight = nn.Parameter(
        weight.to(module.weight.device, torch.int8), requires_grad=False
    )
    if mf_tool.sparsity > 0 and mf_tool.dtypes["low"] != "zero":
        # module.register_parameter(
        #     "lweight", nn.Parameter(
        #         lweight.to(module.weight.device), requires_grad=False
        #     )
        # )
        module.register_parameter(
            "weight_lscale_inv", nn.Parameter(
                lscale.to(module.weight_scale_inv.device, scale_dtype)
            )
        )
        if mask_in_id:
            # [O, NB, BankSize]
            mask = mf_tool.sparse_tool.transform.preprocess(mask)
            idx, idy, idz = torch.where(mask == 1)
            mask_id = idz.view(mask.size(0), mask.size(1), -1)
            module.register_parameter(
                "mask_id", nn.Parameter(
                    mask_id.to(module.weight.device, torch.int8), requires_grad=False
                )
            )
            mask_n = torch.zeros(
                (mask_id.size(0), mask_id.size(1), mf_tool.sparse_tool.bank_size),
                dtype=torch.int8
            )
            mask_n.scatter_(2, mask_id, 1)
            mask_n = mf_tool.sparse_tool.transform.postprocess(mask_n)
            mask = mf_tool.sparse_tool.transform.postprocess(mask)
            assert (mask == mask_n).all()
            module.mask = mask
        else:
            module.register_parameter(
                "mask", nn.Parameter(
                    mask.to(module.weight.device, torch.int8), requires_grad=False
                )
            )
    else:
        # module.lweight = torch.tensor(0, dtype=torch.int8)
        module.weight_lscale_inv = torch.tensor(0, dtype=scale_dtype)
        module.mask = torch.tensor(1, dtype=torch.int8)

    new_weight = module.weight.to(scale_dtype) * \
        module.weight_scale_inv.to(scale_dtype).repeat_interleave(
            mf_tool.bank_size, dim=-1
        ) * mask.to(scale_dtype)

    if mf_tool.sparsity > 0 and mf_tool.dtypes["low"] != "zero":
        new_weight += module.weight.to(scale_dtype) * \
            module.weight_lscale_inv.to(scale_dtype).repeat_interleave(
                mf_tool.bank_size, dim=-1
            ) * (1 - mask).to(scale_dtype)
        
    if smooth and not any([item in name for item in smooth_filters]):
        smooth_scale = tool.module.smooth_scale.clone()
        module.smooth_scale = nn.Parameter(
            smooth_scale.to(module.weight.device), requires_grad=False
        )
        new_weight = new_weight / module.smooth_scale

    diff = ori_weight - new_weight
    print(f"diff mean: {diff.abs().mean():.5f}\tdiff max: {diff.abs().max():.5f}")
    print(f"ori mean: {ori_weight.mean():.5f}\tori max: {ori_weight.max():.5f}")
    print(f"new mean: {new_weight.mean():.5f}\tnew max: {new_weight.max():.5f}")
    
def cuda_hook(
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    kwargs: dict[str, typing.Any],
    device: torch.device
):
    tmp_args_cache = [
        v.to(device) if isinstance(v, torch.Tensor) else v
        for v in args
    ]
    tmp_kwargs_cache = {
        k: v.to(device) if isinstance(v, torch.Tensor) \
            else [
                vi.to(device) if isinstance(vi, torch.Tensor) else vi
                for vi in v
            ] if isinstance(v, tuple) else v
        for k, v in kwargs.items()
    }
    return tmp_args_cache, tmp_kwargs_cache

def reset_past_key_values(past_key_values: DynamicCache, layer_idx: int):
    past_key_values.layers[layer_idx].values = None
    past_key_values.layers[layer_idx].keys = None
    past_key_values.layers[layer_idx].is_initialized = False


class ModelSaver:
    def __init__(self, model: DeepseekV3ForCausalLM, out_dir: str):
        self.model = model
        num_layers = len(model.model.layers)
        self.moe_layers = [
            i for i in range(
                model.config.first_k_dense_replace,
                num_layers,
                model.config.moe_layer_freq
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

    def finish(self):
        with open(
            os.path.join(
                self.out_dir, "model.safetensors.index.json"
            ),
            mode="w"
        ) as fid:
            json.dump(self.param_refs, fid, indent=4, ensure_ascii=True)
        self.model.config.save_pretrained(self.out_dir)
        # with open(
        #     os.path.join(self.out_dir, "config.json"), mode="w"
        # ) as fid:
        #     json.dump(
        #         self.model.config.to_dict(), fid, indent=4, ensure_ascii=True
        #     )

def run(
    model: DeepseekV3ForCausalLM,
    args_cache,
    kwargs_cache,
    device,
    sparsity: dict,
    high_bits: int,
    low_bits: int,
    out_dir,
    smooth=True,
    smooth_filters=[],
    partial_save=False,
    mask_in_id=False,
):
    compress_configs = {
        "experts": get_compress_config(
            sparsity["experts"], high_bits, low_bits
        ),
        "linear": get_compress_config(
            sparsity["linear"], high_bits, low_bits
        )
    }
    if partial_save:
        assert out_dir is not None
        saver = ModelSaver(model, out_dir=out_dir)
        layers = model.model.layers
        model.model.layers = nn.ModuleList([])
        saver.save(model.state_dict(), layer_idx=None)
        model.model.layers = layers
    else:
        saver = None
    
    with torch.no_grad():
        for layer_idx, layer in enumerate(model.model.layers):
            hooks, tools = register_compress_tool(
                layer, compress_configs=compress_configs
            )
            
            if smooth:
                layer = layer.to(device)
                # layer._forward_pre_hooks_with_kwargs = OrderedDict()
                device_hook = layer.register_forward_pre_hook(
                    partial(cuda_hook, device=device), with_kwargs=True
                )
            
                for i in tqdm(
                    range(len(args_cache)),
                    desc=f"Running layer {layer_idx:2d}"
                ):
                    reset_past_key_values(kwargs_cache[i]['past_key_values'], layer_idx)

                    args_cache[i][0] = layer(*args_cache[i], **kwargs_cache[i]).detach().to('cpu')
                    
                    reset_past_key_values(kwargs_cache[i]['past_key_values'], layer_idx)
                device_hook.remove()
                layer = layer.to('cpu')
            
            for hook in hooks.values():
                hook.remove()

            for name, tool in tqdm(tools.items(), desc=f"Solving layer {layer_idx:2d}"):
                tqdm.write(f"### Compressing: {name} ###")
                solve_weight(
                    name,
                    tool,
                    smooth=smooth,
                    smooth_filters=smooth_filters,
                    mask_in_id=mask_in_id,
                )
            # with ProcessPoolExecutor(max_workers=cpu_count()//2) as executor:
            #     futures = list()
            #     for name, tool in tools.items():
            #         futures.append(executor.submit(solve_weight, tool))
            #     # for future in tqdm(as_completed(futures), desc=f"Solving layer {layer_idx:2d}"):
            #     for future in tqdm(futures, desc=f"Solving layer {layer_idx:2d}"):
            #         future.result()
            if partial_save:
                saver.save(
                    layer.state_dict(prefix=f"model.layers.{layer_idx}."), 
                    layer_idx=layer_idx
                )
                model.model.layers[layer_idx] = nn.Module()

    model.config.quantization_config = {
        "activation_scheme": "dynamic",
        "quant_method": "blockwise_int8",
        "mf_linear_config": {
            "__default__": {
                "weight": {
                    "sparsity": sparsity["linear"],
                    "high_bits": 8,
                    "low_bits": low_bits,
                    "mask_in_id": True
                },
                "input": {
                    "sparsity": 0,
                    "high_bits": 8,
                    "low_bits": 0,
                    "mf_format": True
                },
                "smooth": smooth
            },
            ".*experts": {
                "weight": {
                    "sparsity": sparsity["experts"],
                    "high_bits": 8,
                    "low_bits": low_bits,
                    "mask_in_id": True
                },
                "input": {
                    "sparsity": 0,
                    "high_bits": 8,
                    "low_bits": 0,
                    "mf_format": True
                },
                "smooth": smooth
            }
        },
        "weight_block_size": [
            1,
            64
        ]
    }
    model.config.num_hidden_layers = len(model.model.layers)
    if partial_save:
        saver.finish()
    else:
        model.save_pretrained(out_dir, max_shard_size="4GB")

if __name__ == "__main__":
    device = torch.device('cuda:0')
    cache_file = "/ssd01/workspace/sglang-n/exp/data/DeepSeek-V3.1-Terminus/layer00/kiwi_cache.pt"
    args_cache, kwargs_cache = load_cache(cache_file)

    model_name_or_path = "/ssd01/models/DeepSeek-V3.1-Terminus"
    # model_name_or_path = "/ssd01/models/DeepSeek-V3.2-Exp"
    # model = AutoModelForCausalLM.from_pretrained(
    model = DeepseekV3ForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=True
    )
    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    num_layers = 4
    num_samples = 1
    num_layers = len(model.model.layers)
    num_samples = len(args_cache)
    
    model.model.layers = model.model.layers[:num_layers]
    args_cache = args_cache[:num_samples]
    kwargs_cache = kwargs_cache[:num_samples]

    smooth = True
    smooth = False
    smooth_filters = ['experts.']
    smooth_filters = []
    
    sparsity = {
        "linear": 0,
        "experts": 0.875 
    }
    high_bits = 8
    low_bits = 3

    # partial_save = False
    partial_save = True

    mask_in_id = False
    mask_in_id = True

    model_name = os.path.basename(model_name_or_path)
    out_dir = f"/ssd01/workspace/sglang-n/exp/data/{model_name}-model"
    # out_dir = f"/ssd01/models/{model_name}-MF-Int8"
    # out_dir = f"/ssd01/models/{model_name}-MF-W8xH8L3"
    out_dir = f"/ssd01/models/{model_name}-MF-Linear_WInt8-MOE_W8xH8L3"
    if smooth:
        out_dir += "-smooth"
    run(
        model,
        args_cache,
        kwargs_cache,
        device,
        sparsity=sparsity,
        high_bits=high_bits,
        low_bits=low_bits,
        out_dir=out_dir,
        smooth=smooth,
        smooth_filters=smooth_filters,
        partial_save=partial_save,
        mask_in_id=mask_in_id,
    )
    # tokenizer.save_pretrained(out_dir)
    file_list = [
        "configuration.json",
        "configuration_deepseek.py",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "modeling_deepseek.py",
        "inference"
    ]
    for filepath in file_list:
        if os.path.exists(os.path.join(model_name_or_path, filepath)):
            if os.path.isdir(os.path.join(model_name_or_path, filepath)):
                shutil.copytree(
                    os.path.join(model_name_or_path, filepath),
                    os.path.join(out_dir, filepath),
                    dirs_exist_ok=True
                )
            else:
                shutil.copy(
                    os.path.join(model_name_or_path, filepath),
                    os.path.join(out_dir, filepath)
                )
