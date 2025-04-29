from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch
from torch.nn import functional as F
import triton

from sglang.srt.layers.attention import AttentionBackend
from sglang.srt.layers.attention.flashinfer_backend import (
    create_flashinfer_kv_indices_triton,
)
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInfo

import math
from sparseopt.attns.retriever import MFSparseNbits, TokenSparseRetriever
# from sparseopt.utils.plot import plot_3d

from sglang.srt.mem_cache.memory_pool import MFMLATokenToKVPool
from sglang.srt.utils import quantize, debug

import time

class TritonAttnBackend(AttentionBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
    ):
        # Lazy import to avoid the initialization of cuda context
        from sglang.srt.layers.attention.triton_ops.decode_attention import (
            decode_attention_fwd,
        )
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            extend_attention_fwd,
        )

        super().__init__()

        self.decode_attention_fwd = decode_attention_fwd
        self.extend_attention_fwd = extend_attention_fwd

        self.skip_prefill = skip_prefill

        max_bs = model_runner.req_to_token_pool.size

        if kv_indptr_buf is None:
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
        else:
            self.kv_indptr = kv_indptr_buf

        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        if not self.skip_prefill:
            self.qo_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )

            self.mask_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int64, device=model_runner.device
            )

        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens

        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )

        self.num_kv_splits = model_runner.server_args.triton_attention_num_kv_splits
        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1]

        self.forward_metadata = None

        self.max_context_len = model_runner.model_config.context_len

        self.device = model_runner.device

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for triton attention backend."""

        bs = forward_batch.batch_size
        kv_indptr = self.kv_indptr
        spec_info = forward_batch.spec_info

        if forward_batch.forward_mode.is_decode_or_idle():
            if spec_info is None:
                kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = torch.zeros(
                    forward_batch.seq_lens_sum, dtype=torch.int32, device=self.device
                )
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
                bs = kv_indptr.shape[0] - 1

            attn_logits = torch.zeros(
                (
                    bs,
                    self.num_head,
                    self.num_kv_splits,
                    self.v_head_dim + 1,
                ),
                dtype=torch.float32,
                device=self.device,
            )

            qo_indptr = None
            custom_mask = None
            mask_indptr = None
            max_extend_len = None
        elif forward_batch.forward_mode.is_target_verify():
            bs = len(forward_batch.req_pool_indices)
            qo_indptr = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            # Different with flashinfer kv_indptr and kv_indices construction
            kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.zeros(
                kv_indptr[-1], dtype=torch.int32, device=self.device
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            custom_mask = spec_info.custom_mask
            seq_mask_len = self.num_draft_tokens * (
                forward_batch.seq_lens + self.num_draft_tokens
            )
            mask_indptr = self.mask_indptr
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)
            mask_indptr = mask_indptr[: bs + 1]
            max_extend_len = self.num_draft_tokens
            attn_logits = None
        elif forward_batch.forward_mode.is_draft_extend():
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    self.req_to_token,
                )
            )
            mask_indptr = None
            max_extend_len = torch.max(spec_info.accept_length).item()
            attn_logits = None
        else:
            kv_indptr[1 : bs + 1] = torch.cumsum(
                forward_batch.extend_prefix_lens, dim=0
            )
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.zeros(
                forward_batch.extend_prefix_lens.sum().item(),
                dtype=torch.int32,
                device=self.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.extend_prefix_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            qo_indptr = self.qo_indptr
            qo_indptr[1 : bs + 1] = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
            mask_indptr = None
            attn_logits = None
            max_extend_len = torch.max(forward_batch.extend_seq_lens).item()

        self.forward_metadata = (
            attn_logits,
            max_extend_len,
            kv_indptr,
            kv_indices,
            qo_indptr,
            custom_mask,
            mask_indptr,
        )

    def init_cuda_graph_state(
        self, max_bs: int, kv_indices_buf: Optional[torch.Tensor] = None
    ):
        self.cuda_graph_attn_logits = torch.zeros(
            (max_bs, self.num_head, self.num_kv_splits, self.v_head_dim + 1),
            dtype=torch.float32,
            device=self.device,
        )
        if kv_indices_buf is None:
            self.cuda_graph_kv_indices = torch.zeros(
                (max_bs * self.max_context_len),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.cuda_graph_kv_indices = kv_indices_buf

        if not self.skip_prefill:
            self.cuda_graph_custom_mask = torch.zeros(
                (max_bs * self.max_context_len),
                dtype=torch.uint8,
                device=self.device,
            )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
    ):
        assert encoder_lens is None, "Not supported"

        if forward_mode.is_decode_or_idle():
            if spec_info is None:
                kv_indptr = self.kv_indptr
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = self.cuda_graph_kv_indices
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

            attn_logits = self.cuda_graph_attn_logits
            max_extend_len = None
            qo_indptr = None
            custom_mask = None
            mask_indptr = None
        elif forward_mode.is_target_verify():
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            custom_mask = self.cuda_graph_custom_mask
            seq_mask_len = self.num_draft_tokens * (seq_lens + self.num_draft_tokens)
            mask_indptr = self.mask_indptr[: bs + 1]
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)
            max_extend_len = self.num_draft_tokens
            attn_logits = None
        else:
            raise ValueError(
                f"Invalid forward mode: {forward_mode=} for CUDA Graph capture."
            )

        self.forward_metadata = (
            attn_logits,
            max_extend_len,
            kv_indptr,
            kv_indices,
            qo_indptr,
            custom_mask,
            mask_indptr,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
    ):
        # NOTE: encoder_lens expected to be zeros or None
        if forward_mode.is_decode_or_idle():
            # Update kv_indptr, kv_indices
            kv_indptr = self.kv_indptr
            kv_indices = self.cuda_graph_kv_indices
            if spec_info is None:
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens[:bs], dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices[:bs],
                    seq_lens[:bs],
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
            else:
                kv_indptr[: spec_info.kv_indptr.shape[0]] = spec_info.kv_indptr
                kv_indices[: spec_info.kv_indices.shape[0]] = spec_info.kv_indices
        elif forward_mode.is_target_verify():
            # Update qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr
            bs = len(req_pool_indices)
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            custom_mask = self.cuda_graph_custom_mask
            custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
            seq_mask_len = self.num_draft_tokens * (seq_lens + self.num_draft_tokens)
            mask_indptr = self.mask_indptr[: bs + 1]
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)
        else:
            raise ValueError(
                f"Invalid forward mode: {forward_mode=} for CUDA Graph replay."
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def quantize(
        self,
        x: torch.Tensor, # [S, H, D]
        tool: MFSparseNbits,
        qo_indptr: torch.Tensor,
    ):
        if not tool.is_seq_rely:
            return quantize(x, tool)
        for i in range(len(qo_indptr) - 1):
            idx = torch.arange(qo_indptr[i], qo_indptr[i+1])
            x[idx] = quantize(x[idx], tool)
        return x
    
    def prefill_quant(
        self,
        q: torch.Tensor, #[S, H_q, D_a]
        k: torch.Tensor, # [S, H_k, D_a]
        v: torch.Tensor, # [S, H_k, D_v]
        qo_indptr: torch.Tensor, # [B]
        layer: RadixAttention,
    ):
        if "prefill_quant" not in getattr(layer, "modes", []):
            return q, k, v

        assert hasattr(layer, "q_tool") and \
            hasattr(layer, "k_tool") and \
            hasattr(layer, "v_tool"), \
            f"layer {layer.layer_id} does not have q/k/v tools"
            
        time_stamp = time.time()
        q_tool = layer.q_tool
        k_tool = layer.k_tool
        v_tool = layer.v_tool
        # q = q.clone()
        # k = k.clone()
        # v = v.clone()
        # query/key/value quantization
        q = self.quantize(q, q_tool, qo_indptr)
        k = self.quantize(k, k_tool, qo_indptr)
        v = self.quantize(v, v_tool, qo_indptr)
            
        if layer.layer_id == 0:
            time_used = time.time() - time_stamp
            total_time_used = (time.time() - getattr(self, 'time_stamp', time.time())) / getattr(self, 'layer_num', 1)
            debug(
                f'#### q/k/v prefill quant (time used: {time_used:.3f}s / {total_time_used:.3f}s) ####'
                f'\n\tquery\t: {q.shape}'
                f'\n\tkey\t: {k.shape}'
                f'\n\tvalue\t: {v.shape}'
                f'\n\tindptr\t: {qo_indptr.shape}'
                # f'\n\tindptr\t: {qo_indptr}'
                f'\n\tavg_len\t: {qo_indptr[-1].item() / (len(qo_indptr) - 1):.2f}'
            )
            self.time_stamp = time.time()

        return q, k, v

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        (
            _,
            max_extend_len,
            kv_indptr,
            kv_indices,
            qo_indptr,
            custom_mask,
            mask_indptr,
        ) = self.forward_metadata

        if save_kv_cache:
            if isinstance(forward_batch.token_to_kv_pool, MFMLATokenToKVPool):
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v,
                    kv_indptr=kv_indptr,
                    kv_indices=kv_indices,
                    qo_indptr=qo_indptr,
                )
            else:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )

        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        
        # TODO: prefill quant q, k, v
        q, k, v = self.prefill_quant(
            q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k=k,
            v=v,
            qo_indptr=qo_indptr,
            layer=layer,
        )
        # TODO: only for saving kv cache for XAttention
        # if True:
        #     import torch.distributed as dist
        #     if dist.get_rank() == 0:
        #         print(layer.layer_id, len(qo_indptr), len(kv_indices))
        #         if len(qo_indptr) == 2 and len(kv_indices) > 1e5:
        #             import os
        #             data_dir = "/ssd01/workspace/sglang/exp/data/kv_cache"
        #             os.makedirs(data_dir, exist_ok=True)
        #             filepath = os.path.join(data_dir, f"{layer.layer_id:02d}.pth")
        #             print(f'saving data to {filepath}')
        #             torch.save(
        #                 {
        #                     "k_cache": k_cache[kv_indices].cpu(),
        #                     "k": k.cpu(),
        #                     "q": q.cpu(),
        #                     "scaling": layer.scaling,
        #                 },
        #                 filepath
        #             )
        #             if layer.layer_id == 60:
        #                 import sys; sys.exit()

        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k.contiguous(),
            v.contiguous(),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            # forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            k_cache,
            # forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            v_cache,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask,
            mask_indptr,
            max_extend_len,
            layer.scaling,
            layer.logit_cap,
            act_quant="prefill_quant" in getattr(layer, "modes", []),
            x_attn="prefill_x_attn" in getattr(layer, "modes", []),
            verbose=(layer.layer_id == 0),
        )
        # TODO: only for debugging, masked partial head of output
        # self.output_masked(o, layer)
        return o

    def output_masked(self, o: torch.Tensor, layer: RadixAttention):
        # set partial head to zero:
        import torch.distributed as dist
        from sglang.srt.distributed import get_tensor_model_parallel_rank
        # print(dist.get_rank(), o.shape, o.dtype, get_tensor_model_parallel_rank())
        if layer.layer_id == 60 and get_tensor_model_parallel_rank() == 0:
            # o[:, 5*layer.v_head_dim: 6*layer.v_head_dim] = 0
            o[:, 6*layer.v_head_dim: 7*layer.v_head_dim] = 0
    
    def fetch_idx(
        self,
        q: torch.Tensor, # [B, H_q*D_a]
        kv_indptr: torch.Tensor, # [B+1]
        kv_indices: torch.Tensor, # [S]
        token_to_kv_pool: MFMLATokenToKVPool,
        layer: RadixAttention,
    ):
        if "retrieve" not in getattr(layer, "modes", []):
            return kv_indptr, kv_indices
        
        assert isinstance(
            token_to_kv_pool, MFMLATokenToKVPool
        ), f"token_to_kv_pool must be MFMLATokenToKVPool"
        assert hasattr(layer, "retriever"), \
            f"layer {layer.layer_id} does not have retriever"

        time_stamp = time.time()
        retriever = layer.retriever
        k_cache = token_to_kv_pool.get_label_buffer(layer.layer_id)
        if not retriever.active:
            return kv_indptr, kv_indices
        
        # [B, H_q, 1, D]
        q_reverted = q.view(q.size(0), -1, 1, layer.qk_head_dim)

        # TODO: batch forward when retrieving
        # [B]
        seq_len = kv_indptr[1:] - kv_indptr[:-1]
        # [MB]
        seq_idx = torch.where(seq_len > retriever.retain_size)[0]
        if seq_idx.numel() == 0:
            kv_indptr_n, kv_indices_n = kv_indptr, kv_indices
        else:
            # [MB, H_q, 1, D]
            select_q_reverted = q_reverted[seq_idx]
            select_k_cache = [
                k_cache[kv_indices[kv_indptr[i]: kv_indptr[i+1]]]
                for i in seq_idx
            ]
            maxlen = max(len(k) for k in select_k_cache)
            # [MB]
            pad_info = maxlen - seq_len[seq_idx]
            # [MB, H_k, maxlen, D], left paddding
            select_k_cache = torch.stack(
                [
                    F.pad(
                        k, (0, 0, 0, 0, maxlen - len(k), 0),
                        mode="constant", value=0
                    )
                    for k, pad_len in zip(select_k_cache, pad_info)
                ],
                dim=0
            ).transpose(1, 2)
            # [maxlen], which contains torch.tensor([maxlen, maxlen-1, ..., 2, 1, 0])
            range_seq = torch.arange(maxlen, device=select_k_cache.device).flip(dims=[0])
            # [MB, 1, 1, maxlen]
            mask = (range_seq[None, :] < seq_len[seq_idx][:, None])[:, None, None, :]

            # reset retriever
            retriever._reset()
            retriever.key_buffer = select_k_cache
            retriever.seq_len = maxlen
            # [MB, H_k, 1, N] -> [MB, HK, N] -> [MB, N]
            idx = retriever._fetch_idx(select_q_reverted, mask).squeeze(2)[:, 0]
            # remove offset introduced by padding
            idx = idx - pad_info[:, None]
            assert idx.min() >= 0, "idx should be non-negative"
            
            kv_indices_n = [
                kv_indices[kv_indptr[i]: kv_indptr[i+1]] if i not in seq_idx \
                    else kv_indices[idx[seq_idx == i][0] + kv_indptr[i]]
                for i in range(q_reverted.size(0))
            ]
            # set indptr
            kv_indptr_n = kv_indptr.clone()
            kv_indptr_n[0] = 0
            kv_indptr_n[1:] = torch.tensor(
                [len(indice) for indice in kv_indices_n],
                device=kv_indptr.device, dtype=kv_indptr.dtype
            )
            kv_indptr_n = kv_indptr_n.cumsum(dim=0)
            # merge kv_indices
            kv_indices_n = torch.cat(kv_indices_n, dim=-1)
            
            # reset retriever
            retriever._reset()
        
        if layer.layer_id == 0:
            self.layer_num = len(token_to_kv_pool.kv_buffer)
            time_used = time.time() - time_stamp
            total_time_used = (time.time() - getattr(self, 'time_stamp', time.time())) / getattr(self, 'layer_num', 1)
            debug(
                f'#### decode retrieve (time used: {time_used:.3f}s / {total_time_used:.3f}s) ####'
                f'\n\tquery\t: {q.shape}'
                f'\n\tindptr\t: {kv_indptr.shape}'
                # f'\n\tindptr[0]\t: {kv_indptr[0].item()}'
                f'\n\tori\t: {kv_indices.shape}, {kv_indptr[-1].item()}, {kv_indptr[-1].item() / (len(kv_indptr) - 1):.2f}'
                f'\n\tnew\t: {kv_indices_n.shape}, {kv_indptr_n[-1].item()}, {kv_indptr_n[-1].item() / (len(kv_indptr_n) -1):.2f}'
                # f'\n\tscaling\t: {layer.scaling}'
            )
            self.time_stamp = time.time()
        return kv_indptr_n, kv_indices_n

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        attn_logits, _, kv_indptr, kv_indices, _, _, _ = self.forward_metadata

        if save_kv_cache:
            if isinstance(forward_batch.token_to_kv_pool, MFMLATokenToKVPool):
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v,
                    kv_indptr=kv_indptr,
                    kv_indices=kv_indices,
                    qo_indptr=None,
                )
            else:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )
        
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        # TODO: fetch highest-score idx and ptr from kv_indices, kv_indptr
        kv_indptr, kv_indices = self.fetch_idx(
            q, kv_indptr, kv_indices,
            forward_batch.token_to_kv_pool,
            layer,
        )
        
        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            # forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            k_cache,
            # forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            v_cache,
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            kv_indptr,
            kv_indices,
            attn_logits,
            self.num_kv_splits,
            layer.scaling,
            layer.logit_cap,
        )
        # TODO: only for debugging, masked partial head of output
        # self.output_masked(o, layer)
        return o


class TritonMultiStepDraftBackend:
    """
    Wrap multiple triton attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        from sglang.srt.speculative.eagle_utils import generate_draft_decode_kv_indices

        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices
        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.attn_backends = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                TritonAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                )
            )
        self.max_context_len = self.attn_backends[0].max_context_len
        self.device = model_runner.device
        # Cached variables for generate_draft_decode_kv_indices
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]

    def common_template(
        self, forward_batch: ForwardBatch, kv_indices_buffer: torch.Tensor, call_fn: int
    ):
        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        self.generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            forward_batch.req_pool_indices,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.seq_lens,
            kv_indices_buffer,
            self.kv_indptr,
            forward_batch.positions,
            num_seqs,
            self.topk,
            self.pool_len,
            kv_indices_buffer.shape[1],
            self.kv_indptr.shape[1],
            triton.next_power_of_2(num_seqs),
            triton.next_power_of_2(self.speculative_num_steps),
            triton.next_power_of_2(bs),
        )

        for i in range(self.speculative_num_steps):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : seq_lens_sum * self.topk + bs * (i + 1)
            ]
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices = torch.zeros(
            (
                self.speculative_num_steps,
                forward_batch.batch_size * self.topk * self.max_context_len,
            ),
            dtype=torch.int32,
            device=self.device,
        )

        def call_fn(i, forward_batch):
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, kv_indices, call_fn)

    def init_cuda_graph_state(self, max_bs: int):
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, max_bs * self.max_context_len),
            dtype=torch.int32,
            device=self.device,
        )
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, kv_indices_buf=self.cuda_graph_kv_indices[i]
            )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)

    def init_forward_metadata_replay_cuda_graph(self, forward_batch):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                forward_batch.batch_size,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)
