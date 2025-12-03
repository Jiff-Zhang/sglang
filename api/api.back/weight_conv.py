import safetensors
import torch
from torch import nn
from torch.nn import functional as F
import json
import os
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
    compress_config: CompressConfig
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
                tools[name] = get_compress_tool(
                    [module], compress_config=compress_config
                )
            else:
                match_key = list(filter(lambda x: x in name, pair))[0]
                index = pair.index(match_key)
                modules = [
                    layer.get_submodule(name.replace(match_key, item))
                    for item in pair
                ]
                tools[name] = get_compress_tool(
                    modules, compress_config=compress_config
                )
                wo_list = [module.weight.size(0) for module in modules]
                setattr(
                    tools[name], "weight_slice",
                    (sum(wo_list[:index]), sum(wo_list[:index+1]))
                )
                # print(index, match_key, pair, tools[name].weight_slice)
            setattr(tools[name], "ori_module", module)
            hooks[name] = module.register_forward_pre_hook(
                partial(add_batch, tool=tools[name]),
                with_kwargs=True
            )
    return hooks, tools

def solve_weight(name, tool: CompressOPT, smooth=True):
    if "experts." not in name and smooth:
        tool.smooth()
    
    weight = tool.module.weight
    if hasattr(tool, "weight_slice"):
        weight_slice = getattr(tool, "weight_slice")
        weight = weight[weight_slice[0]: weight_slice[1]]

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
        quant_masked=True
    )

    module = tool.ori_module
    ori_weight = get_weight(module)
    
    assert module.weight.size() == weight.size()
    if not isinstance(module, (nn.Linear, FP8Linear)):
        raise NotImplementedError

    # quantize: int8 weight + bfloat16 scale
    # split to high and low
    hweight = mf_tool.sparse_tool.transform.preprocess(weight)
    hweight, lweight, mask = mf_tool.sparse_tool(hweight)
    hweight = mf_tool.sparse_tool.transform.postprocess(hweight)
    lweight = mf_tool.sparse_tool.transform.postprocess(lweight)
    mask = mf_tool.sparse_tool.transform.postprocess(mask)
    # quant high
    hweight = mf_tool.high_quant_tool.transform.preprocess(hweight)
    hweight, scale = mf_tool.high_quant_tool.sym_quant(hweight)
    hweight = mf_tool.high_quant_tool.transform.postprocess(hweight).to(torch.int8)
    hscale = mf_tool.high_quant_tool.transform.postprocess(scale).to(torch.bfloat16)
    # quant low
    if config.sparsity > 0:
        lweight = mf_tool.low_quant_tool.transform.preprocess(lweight)
        lweight, scale = mf_tool.low_quant_tool.sym_quant(lweight)
        lweight = mf_tool.low_quant_tool.transform.postprocess(lweight).to(torch.int8)
        lscale = mf_tool.low_quant_tool.transform.postprocess(scale).to(torch.bfloat16)
    else:
        lweight, lscale = 0, 0

    weight = hweight * mask + lweight * (1 - mask)
    module.weight = nn.Parameter(
        weight.to(module.weight.device), requires_grad=False
    )
    module.weight_scale_inv = nn.Parameter(
        hscale.to(module.weight_scale_inv.device)
    )
    if config.sparsity > 0:
        # module.register_parameter(
        #     "lweight", nn.Parameter(
        #         lweight.to(module.weight.device), requires_grad=False
        #     )
        # )
        module.register_parameter(
            "lweight_scale_inv", nn.Parameter(
                lscale.to(module.weight_scale_inv.device)
            )
        )
        module.register_parameter(
            "mask", nn.Parameter(
                mask.to(module.weight.device, torch.int8), requires_grad=False
            )
        )
    else:
        # module.lweight = torch.tensor(0, dtype=torch.int8)
        module.lweight_scale_inv = torch.tensor(0, dtype=torch.bfloat16)
        module.mask = torch.tensor(1, dtype=torch.int8)

    new_weight = module.weight.to(torch.bfloat16) * \
        module.weight_scale_inv.to(torch.bfloat16).repeat_interleave(
            64, dim=-1
        ) * mask.to(torch.bfloat16)

    if config.sparsity > 0:
        new_weight += module.weight.to(torch.bfloat16) * \
            module.lweight_scale_inv.to(torch.bfloat16).repeat_interleave(
                64, dim=-1
            ) * (1 - mask).to(torch.bfloat16)
        
    if "experts." not in name and smooth:
        smooth_scale = tool.module.smooth_scale
        module.smooth_scale = nn.Parameter(
            smooth_scale.to(smooth_scale.device), requires_grad=False
        )
        new_weight = new_weight / module.smooth_scale

    diff = ori_weight - new_weight
    print(f"diff mean: {diff.abs().mean():.4f}\tdiff max: {diff.abs().max():.4f}")
    print(f"ori mean: {ori_weight.mean():.4f}\tori max: {ori_weight.max():.4f}")
    print(f"new mean: {new_weight.mean():.4f}\tnew max: {new_weight.max():.4f}")
    
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

def run(model, args_cache, kwargs_cache, num_samples, device, compress_config, smooth=True):
    with torch.no_grad():
        for layer_idx, layer in enumerate(model.model.layers):
            hooks, tools = register_compress_tool(
                layer, compress_config=compress_config
            )
            
            layer = layer.to(device)
            # layer._forward_pre_hooks_with_kwargs = OrderedDict()
            device_hook = layer.register_forward_pre_hook(
                partial(cuda_hook, device=device), with_kwargs=True
            )
            
            for i in tqdm(
                range(len(args_cache[:num_samples])),
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
                solve_weight(name, tool, smooth=smooth)
            # with ProcessPoolExecutor(max_workers=cpu_count()//2) as executor:
            #     futures = list()
            #     for name, tool in tools.items():
            #         futures.append(executor.submit(solve_weight, tool))
            #     # for future in tqdm(as_completed(futures), desc=f"Solving layer {layer_idx:2d}"):
            #     for future in tqdm(futures, desc=f"Solving layer {layer_idx:2d}"):
            #         future.result()

if __name__ == "__main__":
    device = torch.device('cuda:0')
    cache_file = "/ssd01/workspace/sglang-n/exp/data/DeepSeek-V3.1-Terminus/layer00/kiwi_cache.pt"
    args_cache, kwargs_cache = load_cache(cache_file)

    model_name_or_path = "/ssd01/models/DeepSeek-V3.1-Terminus"
    model = DeepseekV3ForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        device_map="cpu"
    )
    
    num_layers = 4
    num_samples = 1
    # num_layers = len(model.model.layers)
    # num_samples = len(args_cache)
    model.model.layers = model.model.layers[:num_layers]

    smooth = True
    smooth = False
    
    sparsity = 0
    sparsity = 0.875
    high_bits = 8
    low_bits = 3
    compress_config = get_compress_config(sparsity, high_bits, low_bits)
    
    run(
        model,
        args_cache,
        kwargs_cache,
        num_samples,
        device,
        compress_config=compress_config,
        smooth=smooth
    )

    model.config.quantization_config = {
        "activation_scheme": "dynamic",
        "mf_format": True,
        "quant_method": "blockwise_int8",
        "smooth": smooth,
        "w_sparisty": sparsity,
        "w_low_bits": low_bits,
        "weight_block_size": [
            1,
            64
        ]
    }
    model.config.num_hidden_layers = num_layers
    out_dir = "/ssd01/workspace/sglang-n/exp/data/DeepSeek-V3.1-Terminus-model"
    # out_dir = "/ssd01/models/DeepSeek-V3.1-Terminus-MF-Int8-smooth"
    model.save_pretrained(out_dir, max_shard_size="4GB")
