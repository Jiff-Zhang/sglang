from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_utils import generate_draft_decode_kv_indices
from sglang.srt.utils import (
    get_bool_env_var,
    get_device_core_count,
    get_int_env_var,
    next_power_of_2,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
    is_logging_enabled,
)

from sglang.srt.mem_cache.memory_pool import MFTokenToKVPool
from sglang.srt.mf_tool import MFSparseNbits, TokenSparseRetriever
from sglang.srt.mf_tool import quantize

import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


def logit_capping_mod(logit_capping_method, logit_cap):
    # positive logit_cap -> tanh cap
    if logit_capping_method == "tanh":
        return logit_cap
    else:
        raise ValueError()


@dataclass
class ForwardMetadata:
    attn_logits: torch.Tensor
    attn_lse: torch.Tensor
    max_extend_len: int
    num_kv_splits: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    qo_indptr: torch.Tensor
    custom_mask: torch.Tensor
    mask_indptr: torch.Tensor
    # Sliding window
    window_kv_indptr: torch.Tensor
    window_kv_indices: torch.Tensor
    window_num_kv_splits: torch.Tensor
    window_kv_offsets: torch.Tensor


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

        self.decode_attention_fwd = torch.compiler.disable(decode_attention_fwd)
        self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)

        # Parse args
        self.skip_prefill = skip_prefill
        max_bs = model_runner.req_to_token_pool.size
        self.sliding_window_size = model_runner.sliding_window_size
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        if model_runner.hybrid_gdn_config is not None:
            # For hybrid linear models, layer_id = 0 may not be full attention
            self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()
        else:
            self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[
                -1
            ]
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.device_core_count = get_device_core_count(model_runner.gpu_id)
        self.static_kv_splits = get_bool_env_var(
            "SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS", "false"
        )
        self.max_kv_splits = model_runner.server_args.triton_attention_num_kv_splits

        # Decide whether enable deterministic inference with batch-invariant operations
        self.enable_deterministic = (
            model_runner.server_args.enable_deterministic_inference
        )

        # Configure deterministic inference settings
        if self.enable_deterministic:
            # Use fixed split tile size for batch invariance
            self.split_tile_size = get_int_env_var(
                "SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE", 256
            )
            # Set static_kv_splits to False to use deterministic logic instead
            self.static_kv_splits = False
        else:
            self.split_tile_size = (
                model_runner.server_args.triton_attention_split_tile_size
            )

        if self.split_tile_size is not None:
            self.max_kv_splits = (
                self.max_context_len + self.split_tile_size - 1
            ) // self.split_tile_size

        # Check arguments
        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        # Initialize buffers
        # TODO(Jianan Ji): Make sure it behaves as expected when kv_indptr_buf is provided and sliding window is enabled
        if kv_indptr_buf is None:
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
        else:
            self.kv_indptr = kv_indptr_buf

        # If sliding window is enabled, we might need two sets of buffers
        # because of interleaved attention types (e.g. for Gemma3)
        self.window_kv_indptr = None
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            if kv_indptr_buf is None:
                self.window_kv_indptr = torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
            else:
                # When provided a buffer, create a clone for the second buffer
                self.window_kv_indptr = torch.zeros_like(kv_indptr_buf)

        if not self.skip_prefill:
            self.qo_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )

            self.mask_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int64, device=model_runner.device
            )

        # Initialize forward metadata
        self.forward_metadata: ForwardMetadata = None

        self.cuda_graph_custom_mask = None

    def get_num_kv_splits(
        self,
        num_kv_splits: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        num_token, num_seq = num_kv_splits.shape[0], seq_lens.shape[0]
        # NOTE(alcanderian): Considering speculative_decodeing,
        # num_kv_splits.shape[0] will be topk * real_num_token.
        # And the real_num_token is num_seq in decoding phase.
        num_group = num_token // num_seq

        assert (
            num_group * num_seq == num_token
        ), f"num_seq({num_seq}), num_token({num_token}), something goes wrong!"

        # Legacy dynamic splitting logic (non-deterministic)
        if (
            self.static_kv_splits or self.device_core_count <= 0
        ) and not self.enable_deterministic:
            num_kv_splits.fill_(self.max_kv_splits)
            return

        # deterministic
        if self.split_tile_size is not None and self.enable_deterministic:
            # expand seq_lens to match num_token
            if num_group > 1:
                expanded_seq_lens = seq_lens.repeat_interleave(num_group)
            else:
                expanded_seq_lens = seq_lens

            num_kv_splits[:] = (
                expanded_seq_lens + self.split_tile_size - 1
            ) // self.split_tile_size
            return

        if num_seq < 256:
            SCHEDULE_SEQ = 256
        else:
            SCHEDULE_SEQ = triton.next_power_of_2(num_seq)

        get_num_kv_splits_triton[(1,)](
            num_kv_splits,
            seq_lens,
            num_seq,
            num_group,
            self.num_head,
            self.num_kv_head,
            self.max_kv_splits,
            self.device_core_count,
            MAX_NUM_SEQ=SCHEDULE_SEQ,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for triton attention backend."""

        bs = forward_batch.batch_size
        kv_indptr = self.kv_indptr
        window_kv_indptr = self.window_kv_indptr
        window_kv_indices = None
        window_num_kv_splits = None
        window_kv_offsets = None
        spec_info = forward_batch.spec_info

        if forward_batch.forward_mode.is_decode_or_idle():
            if spec_info is None:
                kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = torch.empty(
                    forward_batch.seq_lens_sum, dtype=torch.int64, device=self.device
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
                # Sliding window
                if (
                    self.sliding_window_size is not None
                    and self.sliding_window_size > 0
                ):
                    window_kv_indptr, window_kv_indices, window_kv_lens, _ = (
                        update_sliding_window_buffer(
                            self.window_kv_indptr,
                            self.req_to_token,
                            self.sliding_window_size,
                            forward_batch.seq_lens,
                            forward_batch.req_pool_indices,
                            bs,
                            self.device,
                            self.token_to_kv_pool_allocator,
                        )
                    )
                    window_num_kv_splits = torch.empty(
                        (bs,), dtype=torch.int32, device=self.device
                    )
                    self.get_num_kv_splits(window_num_kv_splits, window_kv_lens)
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
                bs = kv_indptr.shape[0] - 1

            attn_logits = torch.empty(
                (bs, self.num_head, self.max_kv_splits, self.v_head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            attn_lse = torch.empty(
                (bs, self.num_head, self.max_kv_splits),
                dtype=torch.float32,
                device=self.device,
            )
            num_kv_splits = torch.empty((bs,), dtype=torch.int32, device=self.device)
            self.get_num_kv_splits(num_kv_splits, forward_batch.seq_lens)

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
            kv_indices = torch.empty(
                kv_indptr[-1], dtype=torch.int64, device=self.device
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

            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                # window_kv_offsets is used to calculate the start position in custom mask
                (
                    window_kv_indptr,
                    window_kv_indices,
                    window_kv_lens,
                    window_kv_offsets,
                ) = update_sliding_window_buffer(
                    self.window_kv_indptr,
                    self.req_to_token,
                    self.sliding_window_size,
                    forward_batch.seq_lens,
                    forward_batch.req_pool_indices,
                    bs,
                    self.device,
                    self.token_to_kv_pool_allocator,
                )

            custom_mask = spec_info.custom_mask
            seq_mask_len = self.num_draft_tokens * (
                forward_batch.seq_lens + self.num_draft_tokens
            )
            mask_indptr = self.mask_indptr
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)
            mask_indptr = mask_indptr[: bs + 1]
            max_extend_len = self.num_draft_tokens
            num_kv_splits = None
            attn_logits = None
            attn_lse = None

        elif forward_batch.forward_mode.is_draft_extend():
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    None,
                    self.req_to_token,
                )
            )
            kv_indices = kv_indices.to(torch.int64)
            mask_indptr = None
            # TODO(FIXME): This will trigger an invalid Eagle tree when using
            # `max(spec_info.accept_length_cpu)`.
            # It might have been forgotten to update somewhere.
            max_extend_len = torch.max(spec_info.accept_length).item()
            num_kv_splits = None
            attn_logits = None
            attn_lse = None
        else:
            kv_indptr[1 : bs + 1] = torch.cumsum(
                forward_batch.extend_prefix_lens, dim=0
            )
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                sum(forward_batch.extend_prefix_lens_cpu),
                dtype=torch.int64,
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
            # Sliding window
            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                window_kv_indptr, window_kv_indices, _, _ = (
                    update_sliding_window_buffer(
                        self.window_kv_indptr,
                        self.req_to_token,
                        self.sliding_window_size,
                        forward_batch.extend_prefix_lens,
                        forward_batch.req_pool_indices,
                        bs,
                        self.device,
                        self.token_to_kv_pool_allocator,
                    )
                )

            qo_indptr = self.qo_indptr
            qo_indptr[1 : bs + 1] = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
            mask_indptr = None
            attn_logits = None
            attn_lse = None
            max_extend_len = max(forward_batch.extend_seq_lens_cpu)
            num_kv_splits = None

        self.forward_metadata = ForwardMetadata(
            attn_logits,
            attn_lse,
            max_extend_len,
            num_kv_splits,
            kv_indptr,
            kv_indices,
            qo_indptr,
            custom_mask,
            mask_indptr,
            window_kv_indptr,
            window_kv_indices,
            window_num_kv_splits,
            window_kv_offsets,
        )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
        cuda_graph_num_kv_splits_buf: Optional[torch.Tensor] = None,
    ):
        self.cuda_graph_attn_logits = torch.zeros(
            (max_num_tokens, self.num_head, self.max_kv_splits, self.v_head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.cuda_graph_attn_lse = torch.zeros(
            (max_num_tokens, self.num_head, self.max_kv_splits),
            dtype=torch.float32,
            device=self.device,
        )

        if cuda_graph_num_kv_splits_buf is None:
            self.cuda_graph_num_kv_splits = torch.full(
                (max_num_tokens,),
                self.max_kv_splits,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.cuda_graph_num_kv_splits = cuda_graph_num_kv_splits_buf

        if kv_indices_buf is None:
            self.cuda_graph_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.int64,
                device=self.device,
            )
        else:
            self.cuda_graph_kv_indices = kv_indices_buf

        if not self.skip_prefill:
            self.cuda_graph_custom_mask = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.uint8,
                device=self.device,
            )

        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            if kv_indices_buf is None:
                self.cuda_graph_window_kv_indices = torch.zeros(
                    (max_num_tokens * self.sliding_window_size),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                self.cuda_graph_window_kv_indices = torch.zeros_like(kv_indices_buf)

            self.cuda_graph_window_num_kv_splits = torch.full(
                (max_num_tokens,),
                self.max_kv_splits,
                dtype=torch.int32,
                device=self.device,
            )

            self.cuda_graph_window_kv_offsets = torch.zeros(
                (max_bs,),
                dtype=torch.int32,
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
        spec_info: Optional[SpecInput],
    ):
        assert encoder_lens is None, "Not supported"
        window_kv_indptr = self.window_kv_indptr
        window_kv_indices = None
        window_num_kv_splits = None
        window_kv_offsets = None

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
                if (
                    self.sliding_window_size is not None
                    and self.sliding_window_size > 0
                ):
                    window_kv_indices = self.cuda_graph_window_kv_indices
                    window_num_kv_splits = self.cuda_graph_window_num_kv_splits
                    window_kv_indptr, window_kv_indices, _, _ = (
                        update_sliding_window_buffer_cuda_graph(
                            self.window_kv_indptr,
                            window_kv_indices,
                            self.req_to_token,
                            self.sliding_window_size,
                            seq_lens[:bs],
                            req_pool_indices,
                            bs,
                            self.token_to_kv_pool_allocator,
                        )
                    )
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

            attn_logits = self.cuda_graph_attn_logits
            attn_lse = self.cuda_graph_attn_lse
            max_extend_len = None
            num_kv_splits = self.cuda_graph_num_kv_splits
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

            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                window_kv_indices = self.cuda_graph_window_kv_indices
                window_num_kv_splits = self.cuda_graph_window_num_kv_splits
                window_kv_offsets = self.cuda_graph_window_kv_offsets
                window_kv_indptr, window_kv_indices, _, window_kv_offsets[:bs] = (
                    update_sliding_window_buffer_cuda_graph(
                        self.window_kv_indptr,
                        window_kv_indices,
                        self.req_to_token,
                        self.sliding_window_size,
                        seq_lens[:bs],
                        req_pool_indices,
                        bs,
                        self.token_to_kv_pool_allocator,
                    )
                )

            custom_mask = self.cuda_graph_custom_mask
            custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
            seq_mask_len = self.num_draft_tokens * (seq_lens + self.num_draft_tokens)
            mask_indptr = self.mask_indptr[: bs + 1]
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)
            max_extend_len = self.num_draft_tokens
            num_kv_splits = None
            attn_logits = None
            attn_lse = None
        elif forward_mode.is_draft_extend():
            num_tokens_per_bs = self.speculative_num_steps + 1
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                step=num_tokens_per_bs,
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
            custom_mask = None
            mask_indptr = None
            max_extend_len = num_tokens_per_bs
            num_kv_splits = None
            attn_logits = None
            attn_lse = None
        else:
            raise ValueError(
                f"Invalid forward mode: {forward_mode=} for CUDA Graph capture."
            )

        self.forward_metadata = ForwardMetadata(
            attn_logits,
            attn_lse,
            max_extend_len,
            num_kv_splits,
            kv_indptr,
            kv_indices,
            qo_indptr,
            custom_mask,
            mask_indptr,
            window_kv_indptr,
            window_kv_indices,
            window_num_kv_splits,
            window_kv_offsets,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        # NOTE: encoder_lens expected to be zeros or None
        if forward_mode.is_decode_or_idle():
            # Update kv_indptr, kv_indices
            kv_indptr = self.kv_indptr
            kv_indices = self.cuda_graph_kv_indices
            num_kv_splits = self.cuda_graph_num_kv_splits
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
                num_token = bs
                if (
                    self.sliding_window_size is not None
                    and self.sliding_window_size > 0
                ):
                    window_num_kv_splits = self.cuda_graph_window_num_kv_splits
                    window_kv_indices = self.cuda_graph_window_kv_indices
                    _, _, window_kv_lens, _ = update_sliding_window_buffer_cuda_graph(
                        self.window_kv_indptr,
                        window_kv_indices,
                        self.req_to_token,
                        self.sliding_window_size,
                        seq_lens[:bs],
                        req_pool_indices[:bs],
                        bs,
                        self.token_to_kv_pool_allocator,
                    )
                    self.get_num_kv_splits(
                        window_num_kv_splits[:num_token], window_kv_lens[:bs]
                    )

            else:
                assert False, "Multi-step cuda graph init is not done here."
            self.get_num_kv_splits(num_kv_splits[:num_token], seq_lens[:bs])

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
            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                window_num_kv_splits = self.cuda_graph_window_num_kv_splits
                window_kv_indices = self.cuda_graph_window_kv_indices
                window_kv_offsets = self.cuda_graph_window_kv_offsets
                _, _, window_kv_lens, window_kv_offsets[:bs] = (
                    update_sliding_window_buffer_cuda_graph(
                        self.window_kv_indptr,
                        window_kv_indices,
                        self.req_to_token,
                        self.sliding_window_size,
                        seq_lens[:bs],
                        req_pool_indices,
                        bs,
                        self.token_to_kv_pool_allocator,
                    )
                )
            custom_mask = self.cuda_graph_custom_mask
            custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
            seq_mask_len = self.num_draft_tokens * (seq_lens + self.num_draft_tokens)
            mask_indptr = self.mask_indptr[: bs + 1]
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)
        elif forward_mode.is_draft_extend():
            seq_lens = seq_lens[:bs]
            accept_lens = spec_info.accept_length[:bs]
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[1 : bs + 1] = torch.cumsum(accept_lens, dim=0)
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
        else:
            raise ValueError(
                f"Invalid forward mode: {forward_mode=} for CUDA Graph replay."
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def get_extend_metadata(self, layer):
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            sliding_window_size = (
                layer.sliding_window_size
            )  # Needed for sliding window mask
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
            window_kv_offsets = self.forward_metadata.window_kv_offsets
        else:
            sliding_window_size = -1
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
            window_kv_offsets = None
        return sliding_window_size, kv_indptr, kv_indices, window_kv_offsets

    def kv_cache_transfer(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        layer: RadixAttention,
        cache_func=None,
    ):
        if cache_func is None:
            return k_cache, v_cache, kv_indptr, kv_indices

        time_stamp = time.time()
        k_cache_list = list()
        v_cache_list = list()
        for i in range(len(kv_indptr) - 1):
            k, v = cache_func(
                k_cache[kv_indices[kv_indptr[i] : kv_indptr[i + 1]]],
                v_cache[kv_indices[kv_indptr[i] : kv_indptr[i + 1]]],
            )
            k_cache_list.append(k)
            v_cache_list.append(v)

        k_cache_n = torch.cat(k_cache_list, dim=0)
        v_cache_n = torch.cat(v_cache_list, dim=0)
        kv_indices = torch.arange(
            k_cache_n.size(0), device=kv_indices.device, dtype=kv_indices.dtype
        )
        if is_logging_enabled() and layer.layer_id == 0:
            logger.debug(
                f"<TritonAttnBackend.kv_cache_transfer> "
                f"#time used: {time.time() - time_stamp:.3f}s, "
                f"#kv_indptr.shape: {list(kv_indptr.shape)}, "
                f"#kv_indices.shape: {list(kv_indices.shape)}, "
                f"#ori k_cache.shape: {list(k_cache.shape)}, "
                f"#ori v_cache.shape: {list(v_cache.shape)}, "
                f"#new k_cache.shape: {list(k_cache_n.shape)}, "
                f"#new v_cache.shape: {list(v_cache_n.shape)}, "
            )
        return k_cache_n, v_cache_n, kv_indptr, kv_indices

    def quantize(
        self,
        x: torch.Tensor, # [S, H, D]
        tool: MFSparseNbits,
        indptr: torch.Tensor=None,
    ):
        if not tool.is_seq_rely or indptr is None:
            return quantize(x, tool)
        
        x = x.clone()
        for i in range(len(indptr) - 1):
            idx = torch.arange(indptr[i], indptr[i+1])
            x[idx] = quantize(x[idx], tool)
        return x

    def prefill_quant(
        self,
        q: torch.Tensor, #[S, H_q, D_a]
        k: torch.Tensor, # [S, H_k, D_a]
        v: torch.Tensor, # [S, H_k, D_v]
        qo_indptr: torch.Tensor, # [B]
        k_cache: torch.Tensor, # [CS, H_k, D_a]
        v_cache: torch.Tensor, # [CS, H_k, D_v]
        kv_indptr: torch.Tensor, # [B]
        kv_indices: torch.Tensor, # [S]
        layer: RadixAttention,
    ):
        if "prefill_quant" not in getattr(layer, "modes", []):
            return q, k, v, qo_indptr, k_cache, v_cache, kv_indptr, kv_indices

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

        k_cache = self.quantize(k_cache[kv_indices], k_tool, kv_indptr)
        v_cache = self.quantize(v_cache[kv_indices], v_tool, kv_indptr)
        kv_indices = torch.arange(
            k_cache.size(0), device=kv_indices.device, dtype=kv_indices.dtype
        )
        
        if is_logging_enabled() and layer.layer_id == 0:
            time_used = time.time() - time_stamp
            total_time_used = (time.time() - getattr(self, 'time_stamp', time.time())) / getattr(self, 'layer_num', 1)
            logger.debug(
                f"<TritonAttnBackend.prefill_quant> "
                f"#time used: {time_used:.3f}s / {total_time_used:.3f}s, "
                f"#q.shape: {list(q.shape)}, "
                f"#k.shape: {list(k.shape)}, "
                f"#v.shape: {list(v.shape)}, "
                f"#qo_indptr: {list(qo_indptr.shape)}, "
                f"#avg_len: {qo_indptr[-1].item() / (len(qo_indptr) - 1):.2f}, "
                f"#k_cache.shape: {list(k_cache.shape)}, "
                f"#v_cache.shape: {list(v_cache.shape)}, "
                f"#kv_indptr.shape: {list(kv_indptr.shape)}, "
                f"#kv_indices.shape: {list(kv_indices.shape)}, "
                f"#cache_avg_len: {kv_indptr[-1].item() / (len(kv_indptr) - 1):.2f}, "
            )
            self.time_stamp = time.time()

        return q, k, v, qo_indptr, k_cache, v_cache, kv_indptr, kv_indices

    # Deprecated
    def prefill_retrieve(
        self,
        q: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        forward_batch: ForwardBatch,
        layer: RadixAttention,
    ):
        q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        assert hasattr(layer, 'retriever'), f"layer {layer.layer_id} does not have retriever"
        chunk_size = layer.retriever.topk_chunk_size
        
        # page_table = [0 for _ in range(q.size(0))]
        page_table = list()
        index_table = [[] for _ in range(chunk_size)]
        kv_indptr_table = [[0] for _ in range(chunk_size)]
        kv_indices_table = [[] for _ in range(chunk_size)]
        
        for i, (q_s, k_s) in enumerate(zip(qo_indptr[:-1], kv_indptr[:-1])):
            q_e, k_e = qo_indptr[i+1], kv_indptr[i+1]
            for j in range(1, q_e - q_s + 1):
                # Batch retrieve
                seq_len = k_e - k_s + j
                mod = seq_len % chunk_size
                kv_indices_table[mod].append(kv_indices[k_s: k_e])
                kv_indices_table[mod].append(forward_batch.out_cache_loc[q_s: q_s+j])
                kv_indptr_table[mod].append(kv_indptr_table[mod][-1] + seq_len)
                index_table[mod].append(q_s+j-1)
                
                # Direct retrieve
                # kv_indices_n = torch.cat(
                #     (
                #         kv_indices[k_s: k_e], 
                #         forward_batch.out_cache_loc[q_s: q_s+j]
                #     ),
                #     dim=0
                # )
                # kv_indptr_n = kv_indptr[:2].clone()
                # kv_indptr_n[1] = kv_indices_n.size(0)
                # kv_indptr_n, kv_indices_n, k_cache, v_cache = self.fetch_idx(
                #     q[q_s+j-1: q_s+j], kv_indptr_n, kv_indices_n,
                #     forward_batch.token_to_kv_pool,
                #     layer,
                #     disable_logging=True
                # )
                # page_table.append(kv_indices_n)

        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        for i, (indice, kv_indptr_n, kv_indices_n) in \
                enumerate(zip(index_table, kv_indptr_table, kv_indices_table)):

            if(len(indice) == 0):
                continue
            kv_indptr_n = torch.tensor(
                kv_indptr_n, dtype=kv_indptr.dtype, device=kv_indptr.device
            )
            kv_indices_n = torch.cat(kv_indices_n, dim=0)
            indice = torch.tensor(indice)
            kv_indptr_n, kv_indices_n, k_cache_n, v_cache_n = self.fetch_idx(
                q[indice], kv_indptr_n, kv_indices_n,
                forward_batch.token_to_kv_pool,
                layer,
                disable_logging=True,
                batch=True
            )
            assert k_cache_n.size() == k_cache.size(), f"k_cache cannot be modified"
            assert v_cache_n.size() == v_cache.size(), f"v_cache cannot be modified"
            
            # for j, ind in enumerate(indice):
            #     page_table[ind] = kv_indices_n[kv_indptr_n[j]: kv_indptr_n[j+1]]
            page_table = page_table + [
                (ind, kv_indices_n[kv_indptr_n[j]: kv_indptr_n[j+1]])
                for j, ind in enumerate(indice)
            ]
        assert len(page_table) == q.size(0), f"Page table size mismatch"
        page_table = sorted(page_table, key=lambda x: x[0])
        page_table = [x[1] for x in page_table]

        # padding
        maxlen = max([kv_indices_n.size(0) for kv_indices_n in page_table])
        # TODO: For flash-mla constraint
        D_B_TOPK = 64 * 2
        maxlen = (maxlen + D_B_TOPK - 1) // D_B_TOPK * D_B_TOPK
        page_table = [
            F.pad(
                kv_indices_n,
                (0, maxlen - kv_indices_n.size(0)),
                mode="constant",
                value=-1
            )
            for kv_indices_n in page_table
        ]
        page_table = torch.stack(page_table, dim=0)
        #         kv_indices_n.append(kv_indices[k_s: k_e])
        #         kv_indices_n.append(forward_batch.out_cache_loc[q_s: q_s+j])
        #         kv_indptr_n.append(k_e - k_s + j)
        # kv_indptr_n = torch.tensor(
        #     kv_indptr_n, dtype=kv_indptr.dtype, device=kv_indptr.device
        # )
        # kv_indptr_n = kv_indptr_n.cumsum(0)
        # kv_indices_n = torch.cat(kv_indices_n, dim=0)
        # kv_indptr_n, kv_indices_n, k_cache, v_cache = self.fetch_idx(
        #     q, kv_indptr_n, kv_indices_n,
        #     forward_batch.token_to_kv_pool,
        #     layer,
        # )
        if is_logging_enabled() and layer.layer_id == 0:
            logger.debug(
                f"Prefill retrieve: "
                f"#q.shape: {list(q.shape)}, "
                # f"#kv_indptr_n.shape: {list(kv_indptr_n.shape)}, "
                # f"#kv_indices_n.shape: {list(kv_indices_n.shape)}, "
                f"#page_table.shape: {list(page_table.shape)}, "
                f"#k_cache.shape: {list(k_cache.shape)}, "
                f"#v_cache.shape: {list(v_cache.shape)}, "
            )
        from flash_mla import flash_mla_sparse_fwd
        # TODO: For flash-mla constraint
        B_H = 64
        if q.size(1) % B_H != 0:
            num_to_pad = B_H - q.size(1) % B_H
            q_pad = torch.cat(
                [q] + [q[:, :1].repeat(1, num_to_pad, 1)],
                dim=1
            )
        else:
            q_pad = q
        o, _, _ = flash_mla_sparse_fwd(
            q=q_pad,
            kv=k_cache,
            indices=page_table.unsqueeze(1).to(torch.int32),
            sm_scale=layer.scaling,
            d_v=layer.v_head_dim,
        )
        o = o[:, :q.size(1)]
        return o
    
    def get_verify_buffers_to_fill_after_draft(self):
        """
        Return buffers for verify attention kernels that needs to be filled after draft.

        Typically, these are tree mask and position buffers.
        """
        return [self.cuda_graph_custom_mask, None]

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        pass


    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
        cache_func=None,
    ):
        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        logits_soft_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        # if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
        #     sliding_window_size = (
        #         layer.sliding_window_size
        #     )  # Needed for sliding window mask
        #     kv_indptr = self.forward_metadata.window_kv_indptr
        #     kv_indices = self.forward_metadata.window_kv_indices
        #     window_kv_offsets = self.forward_metadata.window_kv_offsets
        # else:
        #     sliding_window_size = -1
        #     kv_indptr = self.forward_metadata.kv_indptr
        #     kv_indices = self.forward_metadata.kv_indices
        #     window_kv_offsets = None
        sliding_window_size, kv_indptr, kv_indices, window_kv_offsets = \
            self.get_extend_metadata(layer)
        qo_indptr = self.forward_metadata.qo_indptr

        if save_kv_cache:
            if isinstance(forward_batch.token_to_kv_pool, MFTokenToKVPool):
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
        v_cache =forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        k_cache, v_cache, kv_indptr, kv_indices = self.kv_cache_transfer(
            k_cache, v_cache, kv_indptr, kv_indices, 
            layer=layer, cache_func=cache_func
        )

        if 'prefill_retrieve' in getattr(layer, "modes", []):
            # Deprecated
            # o = self.prefill_retrieve(
            #     q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            #     qo_indptr=qo_indptr,
            #     kv_indptr=kv_indptr,
            #     kv_indices=kv_indices,
            #     forward_batch=forward_batch,
            #     layer=layer,
            # )
            q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
            assert hasattr(layer, 'retriever'), f"layer {layer.layer_id} does not have retriever"
            
            label_k_cache = forward_batch.token_to_kv_pool.get_label_buffer(layer.layer_id)
            self.extend_attention_fwd(
                # q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                q.contiguous(),
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
                self.forward_metadata.custom_mask,
                causal,
                self.forward_metadata.mask_indptr,
                self.forward_metadata.max_extend_len,
                layer.scaling,
                logit_cap=logits_soft_cap,
                sliding_window_size=sliding_window_size,
                sinks=sinks,
                window_kv_offsets=window_kv_offsets,
                xai_temperature_len=layer.xai_temperature_len,
                act_quant="prefill_quant" in getattr(layer, "modes", []),
                prefill_retrieve=True,
                prefill_retrieve_config={
                    "k": label_k_cache[forward_batch.out_cache_loc].contiguous(),
                    "k_cache": label_k_cache,
                    "retriever": layer.retriever,
                    "out_cache_loc": forward_batch.out_cache_loc
                },
                is_logging_enabled=is_logging_enabled() and layer.layer_id == 0,
            )
        else:
            q, k, v, qo_indptr, k_cache, v_cache, kv_indptr, kv_indices = \
                self.prefill_quant(
                    q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    k=k,
                    v=v,
                    qo_indptr=qo_indptr,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    kv_indptr=kv_indptr,
                    kv_indices=kv_indices,
                    layer=layer,
                )

            if is_logging_enabled() and layer.layer_id == 0:
                logger.debug(
                    f"<TritonAttnBackend.forward_extend> "
                    f"#window_kv_offsets: {window_kv_offsets}, "
                    f"#q.shape: {list(q.shape)}, "
                    f"#k.shape: {list(k.shape)}, "
                    f"#v.shape: {list(v.shape)}, "
                    f"#k_cache.shape: {list(k_cache.shape)}, "
                    f"#v_cache.shape: {list(v_cache.shape)}, "
                    f"#kv_indptr.shape: {list(kv_indptr.shape)}, "
                    f"#kv_indices.shape: {list(kv_indices.shape)}, "
                    f"#qo_indptr.shape: {list(self.forward_metadata.qo_indptr.shape)}, "
                )

            self.extend_attention_fwd(
                # q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                q.contiguous(),
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
                self.forward_metadata.custom_mask,
                causal,
                self.forward_metadata.mask_indptr,
                self.forward_metadata.max_extend_len,
                layer.scaling,
                logit_cap=logits_soft_cap,
                sliding_window_size=sliding_window_size,
                sinks=sinks,
                window_kv_offsets=window_kv_offsets,
                xai_temperature_len=layer.xai_temperature_len,
                act_quant="prefill_quant" in getattr(layer, "modes", []),
            )
        return o

    def fetch_idx(
        self,
        q: torch.Tensor, # [B, H_q*D_a]
        kv_indptr: torch.Tensor, # [B+1]
        kv_indices: torch.Tensor, # [S]
        token_to_kv_pool: MFTokenToKVPool,
        layer: RadixAttention,
        disable_logging: bool = False,
        batch: bool = False
    ):
        k_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = token_to_kv_pool.get_value_buffer(layer.layer_id)
        if "retrieve" not in getattr(layer, "modes", []):
            return kv_indptr, kv_indices, k_cache, v_cache
        
        assert isinstance(
            token_to_kv_pool, MFTokenToKVPool
        ), f"token_to_kv_pool must be MFTokenToKVPool"
        assert hasattr(layer, "retriever"), \
            f"layer {layer.layer_id} does not have retriever"
        assert hasattr(layer, "num_kv_heads"), \
            f"layer {layer.layer_id} does not have num_kv_heads"
        num_kv_heads = layer.num_kv_heads

        time_stamp = time.time()
        retriever: TokenSparseRetriever = layer.retriever
        label_cache = token_to_kv_pool.get_label_buffer(layer.layer_id)
        if not retriever.active:
            return kv_indptr, kv_indices, k_cache, v_cache
        
        # [B, H_q, 1, D]
        q_reverted = q.view(q.size(0), -1, 1, layer.qk_head_dim)

        # TODO: batch forward when retrieving
        # [B]
        seq_len = kv_indptr[1:] - kv_indptr[:-1]
        # [MB]
        seq_idx = torch.where(seq_len > retriever.retain_size)[0]
        if seq_idx.numel() == 0:
            kv_indptr_n, kv_indices_n = kv_indptr, kv_indices
            k_cache_n, v_cache_n = k_cache, v_cache
        else:
            # if retriever.topk_version in ['v2', 'v2.1', 'v3', 'v3.1']:
            # if True:
            if not batch:
                kv_indices_n = list()
                # for num_kv_heads > 1
                k_cache_n = list()
                v_cache_n = list()
                cum_len = 0
                for i in range(q_reverted.size(0)):
                    retriever._reset()
                    if num_kv_heads == 1:
                        # [L_k, H_k, D] -> [1, L_k, H_k, D] -> [1, H_k, L_k, D]
                        label_cache_cur = label_cache[kv_indices[kv_indptr[i]: kv_indptr[i+1]]][None, ...].transpose(1, 2)
                        # Don't use update because it involves sparse and quantize
                        # retriever.update(label_cache_cur)
                        retriever.key_buffer = label_cache_cur
                        retriever.seq_len = label_cache_cur.size(2)
                        # [1, H_q, 1, N] -> [1, H_q, N]
                        idx = retriever._fetch_idx(q_reverted[i: i+1]).squeeze(2)
                        # TODO: only for DeepSeek MLA, [1, H_q, N] -> [N]
                        idx = idx[0, 0]
                        assert idx.min() >= 0, "idx should be non-negative"
                        kv_indice = kv_indices[idx + kv_indptr[i]]
                    else:
                        assert get_attention_tp_size() == 1, f"Only support one gpu mode, but get {get_attention_tp_size()}"
                        # TODO: fix for MHA, only support one gpu mode
                        # [L_k, H_k, D] -> [1, L_k, H_k, D] -> [1, H_k, L_k, D]
                        k_cache_cur = k_cache[kv_indices[kv_indptr[i]: kv_indptr[i+1]]][None, ...].transpose(1, 2)
                        v_cache_cur = v_cache[kv_indices[kv_indptr[i]: kv_indptr[i+1]]][None, ...].transpose(1, 2)
                        label_cache_cur = label_cache[kv_indices[kv_indptr[i]: kv_indptr[i+1]]][None, ...].transpose(1, 2)
                        # Don't use update because it involves sparse and quantize
                        # retriever.update(label_cache_cur)
                        retriever.key_buffer = label_cache_cur
                        retriever.seq_len = label_cache_cur.size(2)
                        k_cache_cur, v_cache_cur, _ = retriever.fetch_kv(
                            q_reverted[i: i+1],
                            k_cache_cur,
                            v_cache_cur
                        )
                        # [1, H_k, L_k, D] -> [1, L_k, H_k, D] -> [L_k, H_k, D]
                        k_cache_n.append(k_cache_cur.transpose(1, 2)[0])
                        v_cache_n.append(v_cache_cur.transpose(1, 2)[0])
                        kv_indice = torch.arange(
                            k_cache_cur.size(1),
                            device=kv_indices.device,
                            dtype=kv_indices.dtype
                        ) + cum_len
                    cum_len += kv_indice.size(0)
                    kv_indices_n.append(kv_indice)
                retriever._reset()
                
            # TODO: Deprecated
            else:
                assert num_kv_heads == 1, f"Batch mode only support num_kv_heads == 1"
                
                # [MB, H_q, 1, D]
                select_q_reverted = q_reverted[seq_idx]
                # [MB, L_k, H_k, D]
                select_k_cache = [
                    label_cache[kv_indices[kv_indptr[i]: kv_indptr[i+1]]]
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
                # Don't use update because it involves sparse and quantize
                # retriever.update(select_k_cache)
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

            if num_kv_heads == 1:
                k_cache_n = k_cache
                v_cache_n = v_cache
            else:
                k_cache_n = torch.cat(k_cache_n, dim=0)
                v_cache_n = torch.cat(v_cache_n, dim=0)
            
            # reset retriever
            retriever._reset()
        
        if is_logging_enabled() and layer.layer_id == 0 and not disable_logging:
            self.layer_num = len(token_to_kv_pool.get_label_buffer(0))
            time_used = time.time() - time_stamp
            total_time_used = (time.time() - getattr(self, 'time_stamp', time.time())) / getattr(self, 'layer_num', 1)
            ori_info = {
                # "shape": list(kv_indices.shape),
                "total_len": kv_indptr[-1].item(),
                "avg_len": round(kv_indptr[-1].item() / (len(kv_indptr) - 1), 2),
            }
            new_info = {
                # "shape": list(kv_indices_n.shape),
                "total_len": kv_indptr_n[-1].item(),
                "avg_len": round(kv_indptr_n[-1].item() / (len(kv_indptr_n) - 1), 2),
            }
            logger.debug(
                f"<TritonAttnBackend.fetch_idx> "
                f"#time used: {time_used:.3f}s / {total_time_used:.3f}s, "
                f"#q.shape: {list(q.shape)}, "
                f"#kv_indptr.shape: {list(kv_indptr.shape)}, "
                f"#ori kv_indices: {ori_info}, "
                f"#new kv_indices: {new_info}, "
                # f"scaling: {layer.scaling}, "
            )
            self.time_stamp = time.time()
        return kv_indptr_n, kv_indices_n, k_cache_n, v_cache_n
    
    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        logits_soft_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
        else:
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices

        if save_kv_cache:
            if isinstance(forward_batch.token_to_kv_pool, MFTokenToKVPool):
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
        
        # TODO: fetch highest-score idx and ptr from kv_indices, kv_indptr
        kv_indptr, kv_indices, k_cache, v_cache = self.fetch_idx(
            q, kv_indptr, kv_indices,
            forward_batch.token_to_kv_pool,
            layer,
        )

        if is_logging_enabled() and layer.layer_id == 0:
            logger.debug(
                f"<TritonAttnBackend.forward_decode> "
                f"#q.shape: {list(q.shape)}, "
                f"#kv_indptr.shape: {list(kv_indptr.shape)}, "
                f"#kv_indices.shape: {list(kv_indices.shape)}, "
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
            self.forward_metadata.attn_logits,
            self.forward_metadata.attn_lse,
            self.forward_metadata.num_kv_splits,
            self.max_kv_splits,
            layer.scaling,
            logit_cap=logits_soft_cap,
            sinks=sinks,
            xai_temperature_len=layer.xai_temperature_len,
        )
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
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.attn_backends: List[TritonAttnBackend] = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                TritonAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                )
            )
        self.max_context_len = self.attn_backends[0].max_context_len
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.device = model_runner.device
        # Cached variables for generate_draft_decode_kv_indices
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.page_size = model_runner.server_args.page_size

    def common_template(
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: Optional[torch.Tensor],
        call_fn: int,
    ):
        if kv_indices_buffer is None:
            kv_indices_buffer = self.cuda_graph_kv_indices

        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            forward_batch.req_pool_indices,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.seq_lens,
            kv_indices_buffer,
            self.kv_indptr,
            forward_batch.positions,
            self.pool_len,
            kv_indices_buffer.shape[1],
            self.kv_indptr.shape[1],
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps),
            next_power_of_2(bs),
            self.page_size,
        )

        if call_fn is None:
            return

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : seq_lens_sum * self.topk + bs * (i + 1)
            ]
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices = torch.empty(
            (
                self.speculative_num_steps,
                forward_batch.batch_size * self.topk * self.max_context_len,
            ),
            dtype=torch.int64,
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

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, max_num_tokens * self.max_context_len),
            dtype=torch.int64,
            device=self.device,
        )
        self.cuda_graph_num_kv_splits = torch.full(
            (max_num_tokens,),
            self.attn_backends[0].max_kv_splits,
            dtype=torch.int32,
            device=self.device,
        )

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs,
                max_num_tokens,
                kv_indices_buf=self.cuda_graph_kv_indices[i],
                cuda_graph_num_kv_splits_buf=self.cuda_graph_num_kv_splits,
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

        self.common_template(forward_batch, None, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        self.common_template(forward_batch, None, None)

        # NOTE: Multi-step's attention backends use the slice of
        # - kv_indptr buffer (cuda graph and non-cuda graph)
        # - kv_indices buffer (cuda graph only)
        # So we don't need to assign the KV indices inside the attention backend.

        # Compute num_kv_splits only once
        num_token = forward_batch.batch_size * self.topk
        self.attn_backends[-1].get_num_kv_splits(
            self.attn_backends[-1].cuda_graph_num_kv_splits[:num_token],
            forward_batch.seq_lens[:bs],
        )


@triton.jit
def get_num_kv_splits_triton(
    num_kv_splits_ptr,
    seq_lens_ptr,
    num_seq,
    num_group,
    num_head,
    num_kv_head,
    max_kv_splits,
    device_core_count,
    MAX_NUM_SEQ: tl.constexpr,
):
    # TODO: this method is tunable, we need more online serving data to tune it
    offs_seq = tl.arange(0, MAX_NUM_SEQ)
    mask_seq = offs_seq < num_seq

    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=0)
    max_seq_len = tl.max(seq_lens)
    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=max_seq_len)
    min_seq_len = tl.min(seq_lens)
    if max_seq_len * 8 < min_seq_len * 10:
        min_seq_len = max_seq_len
    max_kv_splits_1 = tl.minimum(tl.cdiv(max_seq_len, min_seq_len), max_kv_splits)
    kv_chunk_size_1 = tl.cdiv(max_seq_len, max_kv_splits_1)

    # NOTE: this is a hack to let num_kv_split grows up with seqlen gradually
    ext_seq_len = tl.cast(max_seq_len, tl.float32) / 64.0
    ext_device_core_count = tl.cast(
        device_core_count * tl.maximum(tl.log2(ext_seq_len), 1.0), tl.int32
    )
    block_h, num_kv_group = 16, num_head // num_kv_head
    if num_kv_group == 1:
        token_grid = num_seq * num_group * num_head
    else:
        # from triton_ops/decode_attention.py:_decode_grouped_att_m_fwd
        block_h = tl.minimum(block_h, num_kv_group)
        token_grid = num_seq * num_group * tl.cdiv(num_head, block_h)
    max_kv_splits_2 = tl.minimum(
        tl.cdiv(ext_device_core_count, token_grid), max_kv_splits
    )
    kv_chunk_size_2 = tl.cdiv(max_seq_len, max_kv_splits_2)

    num_kv_splits = tl.maximum(
        tl.cdiv(seq_lens, kv_chunk_size_1), tl.cdiv(seq_lens, kv_chunk_size_2)
    )

    offs_token = offs_seq * num_group
    mask_token = offs_token < num_seq * num_group
    for i in range(0, num_group):
        tl.store(num_kv_splits_ptr + i + offs_token, num_kv_splits, mask=mask_token)


def update_sliding_window_buffer(
    window_kv_indptr,
    req_to_token,
    sliding_window_size,
    seq_lens,
    req_pool_indices,
    bs,
    device,
    token_to_kv_pool_allocator=None,
):
    window_kv_lens = torch.minimum(
        seq_lens,
        torch.tensor(sliding_window_size),
    )
    window_kv_indptr[1 : bs + 1] = torch.cumsum(window_kv_lens, dim=0)
    window_kv_indptr = window_kv_indptr[: bs + 1]
    window_kv_indices = torch.empty(
        window_kv_indptr[-1], dtype=torch.int64, device=device
    )
    window_kv_start_idx = seq_lens - window_kv_lens
    create_flashinfer_kv_indices_triton[(bs,)](
        req_to_token,
        req_pool_indices,
        window_kv_lens,
        window_kv_indptr,
        window_kv_start_idx,
        window_kv_indices,
        req_to_token.stride(0),
    )
    # full to swa index mapping
    if hasattr(token_to_kv_pool_allocator, "translate_loc_from_full_to_swa"):
        kv_last_index = window_kv_indptr[-1]
        window_kv_indices[:kv_last_index] = (
            token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                window_kv_indices[:kv_last_index]
            )
        )
    return window_kv_indptr, window_kv_indices, window_kv_lens, window_kv_start_idx


def update_sliding_window_buffer_cuda_graph(
    window_kv_indptr,
    window_kv_indices,
    req_to_token,
    sliding_window_size,
    seq_lens,
    req_pool_indices,
    bs,
    token_to_kv_pool_allocator=None,
):
    window_kv_lens = torch.minimum(
        seq_lens,
        torch.tensor(sliding_window_size),
    )
    window_kv_indptr[1 : bs + 1] = torch.cumsum(window_kv_lens, dim=0)
    window_kv_indptr = window_kv_indptr[: bs + 1]
    window_kv_start_idx = seq_lens - window_kv_lens
    create_flashinfer_kv_indices_triton[(bs,)](
        req_to_token,
        req_pool_indices,
        window_kv_lens,
        window_kv_indptr,
        window_kv_start_idx,
        window_kv_indices,
        req_to_token.stride(0),
    )
    # full to swa index mapping
    if hasattr(token_to_kv_pool_allocator, "translate_loc_from_full_to_swa"):
        kv_last_index = window_kv_indptr[-1]
        window_kv_indices[:kv_last_index] = (
            token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                window_kv_indices[:kv_last_index]
            )
        )
    return window_kv_indptr, window_kv_indices, window_kv_lens, window_kv_start_idx
