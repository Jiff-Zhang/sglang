import math
import torch

from sparseopt.attns.act_sparse_nbits import MFSparseNbits
from sparseopt.attns.retriever import TokenSparseRetriever

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.dp_attention import is_logging_enabled

def quantize(
    x: torch.Tensor, # [S, H, D]
    tool: MFSparseNbits,
):
    if x.numel() == 0:
        return x
    
    seq_len = x.size(0)
    if tool.is_seq_rely and tool.bank_size > 0:
        quant_len = math.floor(seq_len / tool.bank_size) * tool.bank_size
    else:
        quant_len = seq_len
    return torch.cat(
        [
            tool(x[:quant_len].transpose(0, -2)).transpose(0, -2),
            x[quant_len:]
        ],
        dim=0
    )

def register_mf_tool(layer: RadixAttention, num_kv_heads: int):
    assert isinstance(layer, RadixAttention), f"layer must be an instance of RadixAttention, got {type(layer)}"
    assert num_kv_heads > 0, f"num_kv_heads must be > 0, got {num_kv_heads}"
    
    # return
    bank_size = 64
    modes = [
        # "prefill_quant",    # whether to quantize key/value/query/score when prefill
        # "cache_quant",      # whether to quantize cached key & value
        # "retrieve",         # whether to quantize cached key and retrieve tokens when decode
    ]

    per_bank_tool = MFSparseNbits(
        bank_size=bank_size,
        sparsity=0.,
        quant_mode="per_bank",
        dtypes={"high": "int8", "low": "zero"},
        quant_symmetric=True,
        quant_masked=True,
    )
    per_group_tool = MFSparseNbits(
        bank_size=bank_size,
        sparsity=0.,
        quant_mode="per_group",
        dtypes={"high": "int8", "low": "zero"},
        quant_symmetric=True,
        quant_masked=True,
    )
    per_block_tool = MFSparseNbits(
        bank_size=bank_size,
        sparsity=0.,
        quant_mode="per_block",
        dtypes={"high": "int8", "low": "zero"},
        quant_symmetric=True,
        quant_masked=True,
    )

    q_tool = per_bank_tool
    k_tool = per_bank_tool
    v_tool = per_group_tool
    k_cache_tool = per_bank_tool
    v_cache_tool = per_group_tool
    
    retriever = TokenSparseRetriever(
        active=True,
        # retain_size=2048,
        # retain_size=4096,
        retain_size=8192,
        chunk_size=1,
        recent_size=64,
        bank_size=bank_size,
        sparsity=0.875,
        dtypes={"high": "int4", "low": "zero"},
        sparse_mode="per_bank",
        quant_mode="per_bank",
        q_dtypes={"high": "int8", "low": "zero"},
        q_sparse_mode="per_bank",
        q_quant_mode="per_bank",
        # share=False,
        share=True,
        share_mode="mean",
        # share_mode="max",
        mean_trick=False,
        # mean_trick=True,
        # softmax_scale=False,
        softmax_scale=True,
        softmax_version="v0",
        # topk_version='v0',
        topk_version='v2',
        # topk_version='v2.1',
        # topk_version='v2.2',
        # topk_version='v2.3',
        topk_chunk_size=64,
        # # topk_version='v3',
        # # topk_version='v3.1',
        # topk_version='v3.2',
        # # topk_mini_k=256,
        # topk_mini_k=-1,
        # topk_chunk_size=4096,
        all_reduce=num_kv_heads == 1,
        auto_reset=False,
        qk_scaling=layer.scaling, # TODO: unmark code for scaling
    )
    setattr(layer, "modes", modes)
    setattr(layer, "q_tool", q_tool)
    setattr(layer, "k_tool", k_tool)
    setattr(layer, "v_tool", v_tool)
    setattr(layer, "k_cache_tool", k_cache_tool)
    setattr(layer, "v_cache_tool", v_cache_tool)
    setattr(layer, "retriever", retriever)
    setattr(layer, "num_kv_heads", num_kv_heads)
    if is_logging_enabled() and layer.layer_id == 0:
        print(f"### modes ###")
        print(modes)
        print(f"### retriever ###")
        retriever.print_stats()
        print(f"### q_tool ###")
        q_tool.print_stats()
        print(f"### k_tool ###")
        k_tool.print_stats()
        print(f"### v_tool ###")
        v_tool.print_stats()
        print(f"### k_cache_tool ###")
        k_cache_tool.print_stats()
        print(f"### v_cache_tool ###")
        v_cache_tool.print_stats()