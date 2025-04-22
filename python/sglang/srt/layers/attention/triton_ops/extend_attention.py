# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for prefill.
It supports page size = 1 and prefill with KV cache (i.e. extend).
"""

import torch
import triton
import triton.language as tl
from einops import rearrange

from sglang.srt.layers.attention.triton_ops.prefill_attention import (
    context_attention_fwd,
)
from sglang.srt.utils import is_hip, debug, check

is_cuda_available = torch.cuda.is_available()
if is_cuda_available:
    CUDA_CAPABILITY = torch.cuda.get_device_capability()

is_hip_ = is_hip()

@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def quantize_2d(
    x,
    num_bits: tl.constexpr,
    bank_size: tl.constexpr,
    dim: tl.constexpr,
    magic_num: tl.constexpr=1.0,
    active: tl.constexpr=False,
):
    if not active:
        return x
    
    # TODO: quant attention weights
    tl.static_print(active, num_bits, bank_size, dim, x.shape)
    # tl.device_print('max', tl.max(x))
    # x_p = tl.zeros_like(x)
    
    if dim != 1:
        x = tl.trans(x, dim, 1)

    # reshape to bank_size
    x = tl.reshape(
        x, 
        (x.shape[0], x.shape[1] // bank_size, bank_size)
    )
    
    # symmetric quantization
    max_ranges = tl.max(tl.abs(x), axis=2, keep_dims=True) * magic_num
    max_int = 2 ** (num_bits - 1) - 1
    scales = max_ranges / max_int if num_bits > 1 else max_ranges
    # scales = max_ranges / (max_int + 1) if self.num_bits > 1 else max_ranges

    # set num less than smallest number to smallest number to avoid overflow
    # smallest_normal = torch.finfo(x.dtype).smallest_normal * max_ranges.clamp(1) 
    # scales = torch.maximum(scales, smallest_normal)
    # scales[scales < smallest_normal] = smallest_normal
    # scales[scales == 0] = 1
    scales = tl.where(scales == 0, 1, scales)
    
    # quant
    x = tl.clamp(
        tl.div_rn(x, scales),
        -max_int if num_bits > 1 else -1,
        max_int
    )
    x = x * scales

    # revert shape
    x = tl.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
    
    if dim != 1:
        x = tl.trans(x, dim, 1)
    # return tl.reshape(x, x_p.shape)
    return x

@triton.jit
def _fwd_kernel(
    Q_Extend,
    K_Extend,
    V_Extend,
    QK_Extend,
    O_Extend,
    K_Buffer,
    V_Buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    mask_ptr,
    mask_indptr,
    sm_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_qkbs,
    stride_qkh,
    stride_qkq,
    stride_obs,
    stride_oh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    logit_cap: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: tl.constexpr,
    STORE_TRANSPOSE: tl.constexpr,
    ACT_QUANT: tl.constexpr,
    SAVE_QK: tl.constexpr,
    USE_QK_MASKED: tl.constexpr,
    VERBOSE: tl.constexpr,
    MAX_IDX: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start_idx
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_len_prefix = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
    cur_seq_len = cur_seq_len_prefix + cur_seq_len_extend

    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend

    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    offs_q = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0
    )

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        offs_qpe = (
            (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
            * stride_qbs
            + cur_head * stride_qh
            + offs_dpe[None, :]
        )
        qpe = tl.load(Q_Extend + offs_qpe, mask=mask_m[:, None], other=0.0)

    # stage 1: compute scores with prefix
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    for start_n in range(0, cur_seq_len_prefix, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_len_prefix
        offs_kv_loc = tl.load(
            kv_indices + cur_seq_kv_start_idx + start_n + offs_n, mask=mask_n, other=0
        )

        # load k in transposed way
        offs_buf_k = (
            offs_kv_loc[None, :] * stride_buf_kbs
            + cur_kv_head * stride_buf_kh
            + offs_d[:, None]
        )
        k = tl.load(
            K_Buffer + offs_buf_k, mask=(mask_n[None, :]) & (mask_d[:, None]), other=0.0
        )

        qk = tl.dot(q.to(k.dtype), k)
        if BLOCK_DPE > 0:
            offs_kpe = (
                offs_kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_dpe[:, None]
            )
            kpe = tl.load(
                K_Buffer + offs_kpe,
                mask=mask_n[None, :],
                other=0.0,
            )
            qk += tl.dot(qpe.to(kpe.dtype), kpe)
        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        if USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_len
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            custom_mask &= mask_m[:, None] & mask_n[None, :]
            qk = tl.where(custom_mask, qk, float("-inf"))
        else:
            qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

        # TODO: save qk
        offs_qk = (
            cur_seq * stride_qkbs
            + cur_head * stride_qkh
            + (cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qkq
            + (start_n + offs_n[None, :])
        )
        # DEBUG
        # if tl.min(offs_qk) < 0:
        #     tl.device_print('offset-prefill-min-overflow', tl.min(offs_qk))
        if tl.max(offs_qk) >= MAX_IDX:
            tl.device_print('offset-prefill-max-overflow', tl.max(offs_qk))
        if SAVE_QK:
            tl.store(
                QK_Extend + offs_qk,
                qk,
                mask=(mask_m[:, None] & mask_n[None, :]),
            )
        elif USE_QK_MASKED:
            qk_mask = tl.load(
                QK_Extend + offs_qk,
                mask=(mask_m[:, None] & mask_n[None, :]),
            )
            # qk = tl.where(qk_mask != float("-inf"), qk, float("-inf"))
            qk = tl.where(qk_mask != float("-inf"), qk, -2 ** 50)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        offs_buf_v = (
            offs_kv_loc[:, None] * stride_buf_vbs
            + cur_kv_head * stride_buf_vh
            + offs_dv[None, :]
        )
        v = tl.load(
            V_Buffer + offs_buf_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0
        )
        # TODO: quantize p (attention weights)
        p = quantize_2d(p, num_bits=8, bank_size=64, dim=1, active=ACT_QUANT)
        p = p.to(v.dtype)
        acc = acc * re_scale[:, None] + tl.dot(p, v)

        e_max = n_e_max

    # stage 2: compute the triangle part

    cur_block_m_end = tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
    for start_n in range(0, cur_block_m_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_block_m_end

        # load k in transposed way
        offs_k = (
            (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
            + cur_kv_head * stride_kh
            + offs_d[:, None]
        )
        k = tl.load(
            K_Extend + offs_k, mask=(mask_n[None, :]) & (mask_d[:, None]), other=0.0
        )

        qk = tl.dot(q, k, out_dtype=tl.float32)
        if BLOCK_DPE > 0:
            offs_kpe = (
                (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                + cur_kv_head * stride_kh
                + offs_dpe[:, None]
            )
            kpe = tl.load(
                K_Extend + offs_kpe,
                mask=mask_n[None, :],
                other=0.0,
            )
            qk += tl.dot(qpe, kpe)

        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_len
                + cur_seq_len_prefix
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            custom_mask &= mask_m[:, None] & mask_n[None, :]
            qk = tl.where(custom_mask, qk, float("-inf"))
        else:
            mask_causual = (cur_block_m * BLOCK_M + offs_m[:, None]) >= (
                start_n + offs_n[None, :]
            )
            mask_causual &= mask_m[:, None] & mask_n[None, :]
            qk = tl.where(mask_causual, qk, float("-inf"))
            
        # TODO: save qk
        offs_qk = (
            cur_seq * stride_qkbs
            + cur_head * stride_qkh
            + (cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qkq
            + (cur_seq_len_prefix + start_n + offs_n[None, :])
        )
        # DEBUG
        # if tl.min(offs_qk) < 0:
        #     tl.device_print('offset-extend-min-overflow', tl.min(offs_qk))
        if tl.max(offs_qk) >= MAX_IDX:
            tl.device_print('offset-extend-max-overflow', tl.max(offs_qk))
        if SAVE_QK:
            tl.store(
                QK_Extend + offs_qk,
                qk,
                mask=(mask_m[:, None] & mask_n[None, :]),
            )
        elif USE_QK_MASKED:
            qk_mask = tl.load(
                QK_Extend + offs_qk,
                mask=(mask_m[:, None] & mask_n[None, :]),
            )
            # qk = tl.where(qk_mask != float("-inf"), qk, float("-inf"))
            qk = tl.where(qk_mask != float("-inf"), qk, -2 ** 50)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        offs_v = (
            (cur_seq_extend_start_idx + start_n + offs_n[:, None]) * stride_vbs
            + cur_kv_head * stride_vh
            + offs_dv[None, :]
        )
        v = tl.load(
            V_Extend + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0
        )
        # TODO: quantize p (attention weights)
        p = quantize_2d(p, num_bits=8, bank_size=64, dim=1, active=ACT_QUANT)
        p = p.to(v.dtype)
        acc = acc * re_scale[:, None] + tl.dot(p, v)

        e_max = n_e_max

    offs_o = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    if STORE_TRANSPOSE:
        tl.store(
            O_Extend + offs_o.T,
            (acc / deno[:, None]).T,
            mask=(mask_m[:, None] & mask_dv[None, :]).T,
        )
    else:
        tl.store(
            O_Extend + offs_o,
            acc / deno[:, None],
            mask=mask_m[:, None] & mask_dv[None, :],
        )

def extend_attention_fwd(
    q_extend, # [Le, Hq, Da]
    k_extend, # [Le, Hk, Da]
    v_extend, # [Le, Hk, Dv]
    o_extend, # [Le, Hk, Dv]
    k_buffer, # [Bf, Hk, Da]
    v_buffer, # [Bf, Hk, Dv]
    qo_indptr, # [B + 1]
    kv_indptr, # [B + 1]
    kv_indices, # [Lp]
    custom_mask,
    mask_indptr,
    max_len_extend,
    sm_scale=None,
    logit_cap=0.0,
    skip_prefix_custom_mask=True,
    act_quant=False,
    x_attn=False,
    verbose=False,
):
    """
    q_extend, k_extend, v_extend, o_extend: contiguous tensors

    k_buffer, v_buffer: (prefix + extend) tensors in mem_manager
    """
    Lq, Lk, Lv = (
        q_extend.shape[-1],
        k_extend.shape[-1],
        v_extend.shape[-1],
    )

    if Lq == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lq == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    elif Lq == 192:
        BLOCK_DMODEL = 128
        BLOCK_DPE = 64
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lq)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    if is_hip_:
        BLOCK_M, BLOCK_N = (64, 64)
        num_warps = 4

    else:
        if is_cuda_available and CUDA_CAPABILITY[0] >= 9:
            if Lq <= 256:
                BLOCK_M, BLOCK_N = (128, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 64)
        elif is_cuda_available and CUDA_CAPABILITY[0] >= 8:
            if Lq <= 128:
                BLOCK_M, BLOCK_N = (128, 128)
            elif Lq <= 256:
                BLOCK_M, BLOCK_N = (64, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 64)
        else:
            BLOCK_M, BLOCK_N = (64, 64) if Lq <= 128 else (32, 32)

        num_warps = 4 if Lk <= 64 else 8

    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    batch_size, head_num = qo_indptr.shape[0] - 1, q_extend.shape[1]
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    USE_CUSTOM_MASK = custom_mask is not None
    # Skip custom mask for prefix part
    SKIP_PREFIX_CUSTOM_MASK = skip_prefix_custom_mask

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    num_stages = 1

    extra_kargs = {}
    if is_hip_:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}

    # debug(
    #     q_extend.shape, k_extend.shape, v_extend.shape, o_extend.shape, k_buffer.shape, v_buffer.shape, grid, mask_indptr, custom_mask
    # )

    x_attn = True
    if x_attn:
        # [B]
        len_extend = qo_indptr[1:] - qo_indptr[:-1]
        len_prefix = kv_indptr[1:] - kv_indptr[:-1]
        # # [B]
        # num_attn_nodes = len_extend * (len_extend + len_prefix) * head_num
        # debug(num_attn_nodes)
        # # [1]
        # num_attn_nodes = num_attn_nodes.to(torch.int64).sum()
        # debug(num_attn_nodes)
        qk_extend = o_extend.new_ones(
            (
                batch_size,  head_num,
                triton.cdiv(max_len_extend, BLOCK_M) * BLOCK_M,
                triton.cdiv((len_prefix + len_extend).max(), BLOCK_N) * BLOCK_N
            )
        ) * float("-inf")
        qk_extend = qk_extend.contiguous()
        if verbose:
            debug('qk_extend', qk_extend.shape, qk_extend.numel(), qk_extend.device, qk_extend.dtype)
        stage_kwargs_list = [
            {
                "SAVE_QK": True,
                "USE_QK_MASKED": False,
            },
            {
                "SAVE_QK": False,
                "USE_QK_MASKED": True,
            }
        ]
    else:
        qk_extend = None
        stage_kwargs_list = [
            {
                "SAVE_QK": False,
                "USE_QK_MASKED": False,
            }
        ]

    import torch.distributed as dist
    for stage_kwargs in stage_kwargs_list:
        _fwd_kernel[grid](
            q_extend,
            k_extend,
            v_extend,
            qk_extend,
            o_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask,
            mask_indptr,
            sm_scale,
            kv_group_num,
            q_extend.stride(0),
            q_extend.stride(1),
            k_extend.stride(0),
            k_extend.stride(1),
            v_extend.stride(0),
            v_extend.stride(1),
            qk_extend.stride(0) if qk_extend is not None else 0,
            qk_extend.stride(1) if qk_extend is not None else 0,
            qk_extend.stride(2) if qk_extend is not None else 0,
            o_extend.stride(0),
            o_extend.stride(1),
            k_buffer.stride(0),
            k_buffer.stride(1),
            v_buffer.stride(0),
            v_buffer.stride(1),
            logit_cap=logit_cap,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DPE=BLOCK_DPE,
            BLOCK_DV=BLOCK_DV,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            Lq=Lq,
            Lv=Lv,
            USE_CUSTOM_MASK=USE_CUSTOM_MASK,
            SKIP_PREFIX_CUSTOM_MASK=SKIP_PREFIX_CUSTOM_MASK,
            STORE_TRANSPOSE=is_hip_,
            num_warps=num_warps,
            num_stages=num_stages,
            ACT_QUANT=act_quant,
            # SAVE_QK=save_qk,
            # USE_QK_MASKED=use_qk_masked,
            **stage_kwargs,
            # VERBOSE=verbose and dist.get_rank() == 0,
            VERBOSE=False,
            MAX_IDX=qk_extend.numel() if qk_extend is not None else float("inf"),
            **extra_kargs,
        )
        if x_attn and stage_kwargs["SAVE_QK"]:
            if verbose:
                debug('before', (qk_extend != float("-inf")).float().mean().item())
            try:
                qk_extend = adjust_x_attn_qk(
                    qk_extend, stride=8, block=16, threshold=0.9, verbose=verbose
                )
            except torch.OutOfMemoryError as e:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                qk_extend = adjust_x_attn_qk(
                    qk_extend, stride=8, block=16, threshold=0.9, verbose=verbose
                )
            if verbose:
                debug('after', (qk_extend != float("-inf")).float().mean().item())
    # if verbose:
    #     debug(qk_extend)
    check(o_extend)

def adjust_x_attn_qk(
    qk_extend: torch.Tensor, # [B, H, Q, K]
    stride=8,
    block=16,
    threshold=0.9,
    verbose=False,
):
    assert block % stride == 0, f"block: {block}, stride: {stride}"
    B, H, Q, K = qk_extend.shape
    Q_A = Q // block * block
    K_A = K // block * block

    # [B, H, Q_A, K_A] -> [B, H, qns, kns, q_s, k_s]
    folded_qk = rearrange(
        qk_extend[:, :, :Q_A, :K_A],
        "b h (qns q_s) (kns k_s) -> b h qns kns q_s k_s",
        q_s=stride,
        k_s=stride,
    )
    # flip k_s
    folded_qk = folded_qk.flip(dims=[-1])
    # calculate diagonal mean
    # [B, H, qns, kns, q_s, k_s]
    mask = torch.logical_and(
        torch.eye(stride, device=qk_extend.device, dtype=torch.bool),
        folded_qk != float("-inf")
    )
    # [B, H, qns, kns]
    num_nonzero = mask.to(folded_qk.dtype).sum(dim=-1).sum(dim=-1)
    # [B, H, qns, kns, q_s, k_s] -> [B, H, qns, kns]
    mean_qk = torch.where(mask, folded_qk, 0).sum(dim=-1).sum(dim=-1) / \
        num_nonzero.clamp(min=1)
    # might cause nan when all elements are -inf
    # mean_qk = torch.where(num_nonzero != 0, mean_qk, float("-inf"))
    mean_qk = torch.where(num_nonzero != 0, mean_qk, torch.finfo(mean_qk.dtype).min)
    # if verbose:
    #     debug("sum", torch.where(mask, folded_qk, 0).sum(dim=-1).sum(dim=-1), "num", mask.sum(dim=-1).sum(dim=-1).clamp(min=1), mask.shape, folded_qk.shape)

    # [B, H, qns, kns] -> [B, H, qnb, knb*q_s_b*k_s_b]
    mean_qk = rearrange(
        mean_qk,
        "b h (qnb q_s_b) (knb k_s_b) -> b h qnb (knb q_s_b k_s_b)",
        q_s_b=block // stride,
        k_s_b=block // stride,
    )
    # [B, H, qnb, knb*q_s_b*k_s_b] -> [B, H, qnb, knb]
    scores = rearrange(
        mean_qk.softmax(dim=-1),
        "b h qnb (knb q_s_b k_s_b) -> b h qnb knb (q_s_b k_s_b)",
        q_s_b=block // stride,
        k_s_b=block // stride,
    ).sum(dim=-1)
    # sort
    sort_scores, sort_idx = torch.sort(scores, dim=-1, descending=True)
    cum_scores = torch.cumsum(sort_scores, dim=-1)
    # [B, H, qnb, knb] -> [B, H, qnb]
    retain_len = ((cum_scores < threshold).int().sum(dim=-1) + 1).clamp(max=scores.size(-1))
    # [knb]
    range_block = torch.arange(sort_idx.size(-1), device=retain_len.device)
    # [B, H, qnb, knb]
    mask = range_block[None, None, None, :] < retain_len[..., None]
    sort_idx = torch.where(mask, sort_idx, sort_idx[..., :1].repeat(1, 1, 1, sort_idx.size(-1)))
    mask.zero_()
    mask.scatter_(dim=-1, index=sort_idx, value=True)
    # if verbose:
    #     debug(retain_len, 'scores', scores, 'mean', mean_qk) #, 'softmax', mean_qk.softmax(dim=-1), 'fold', folded_qk)
    #     debug(mask.int().sum(-1) == retain_len)
    
    # [B, H, Q_A, K_A] -> [B, H, qnb, knb, q_b, k_b]
    folded_qk = rearrange(
        qk_extend[:, :, :Q_A, :K_A],
        "b h (qnb q_b) (knb k_b) -> b h qnb knb q_b k_b",
        q_b=block,
        k_b=block,
    )
    folded_qk = torch.where(mask[..., None, None], folded_qk, float("-inf"))
    # revert [B, H, qnb, knb, q_b, k_b] -> [B, H, Q_A, K_A]
    qk_extend[:, :, :Q_A, :K_A] = rearrange(
        folded_qk,
        "b h qnb knb q_b k_b -> b h (qnb q_b) (knb k_b)",
    )
    
    # if verbose:
    #     debug(Q_A, K_A, mean_qk.shape, scores.shape, retain_len.float().mean() / scores.size(-1), mask.float().mean())
    
    return qk_extend

def redundant_attention(
    q_extend,
    o_extend,
    k_buffer,
    v_buffer,
    b_req_idx,
    b_start_loc,
    b_seq_len,
    b_seq_len_prefix,
    max_len_in_batch,
):
    total_token_num = k_buffer.shape[0]
    B, H_Q, D = b_req_idx.shape[0], q_extend.shape[-2], q_extend.shape[-1]
    q_buffer = torch.empty(
        (total_token_num, H_Q, D), dtype=q_extend.dtype, device=q_extend.device
    )

    pt = 0
    for i in range(B):
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]
        pl, pr = b_start_loc[i] + b_seq_len_prefix[i], b_start_loc[i] + b_seq_len[i]
        q_buffer[pl:pr] = q_extend[pt : pt + cur_seq_len_extend]
        pt += cur_seq_len_extend

    o_buffer = torch.empty_like(q_buffer)
    context_attention_fwd(
        q_buffer, k_buffer, v_buffer, o_buffer, b_start_loc, b_seq_len, max_len_in_batch
    )

    pt = 0
    for i in range(B):
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]
        pl, pr = b_start_loc[i] + b_seq_len_prefix[i], b_start_loc[i] + b_seq_len[i]
        o_extend[pt : pt + cur_seq_len_extend] = o_buffer[pl:pr]
        pt += cur_seq_len_extend
