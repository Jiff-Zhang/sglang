# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/quantization/fp8.py

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from torch.nn import Module
from einops import rearrange

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.parameter import BlockQuantScaleParameter, RowvLLMParameter, _ColumnvLLMParameter, ModelWeightParameter
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.int8_utils import apply_w8a8_block_int8_linear
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.utils import set_weight_attrs
from sglang.srt.mf_tool import generate_mask

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = logging.getLogger(__name__)


class BlockInt8Config(QuantizationConfig):
    """Config class for INT8."""

    def __init__(
        self,
        is_checkpoint_int8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: List[int] = None,
        smooth: bool = False,
        mf_format: bool = False,
        w_sparsity: float = 0.0,
        mask_in_id: bool = False
    ) -> None:
        self.is_checkpoint_int8_serialized = is_checkpoint_int8_serialized
        if is_checkpoint_int8_serialized:
            logger.warning(
                "Detected int8 checkpoint. Please note that the "
                "format is experimental and subject to change."
            )
        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        if weight_block_size is not None:
            if not is_checkpoint_int8_serialized:
                raise ValueError(
                    f"The block-wise quantization only supports int8-serialized checkpoint for now."
                )
            if len(weight_block_size) != 2:
                raise ValueError(
                    f"The quantization block size of weight must have 2 dimensions, but got {len(weight_block_size)} dimensions."
                )
            if activation_scheme != "dynamic":
                raise ValueError(
                    f"The block-wise quantization only supports dynamic activation scheme for now, but got {activation_scheme} activation scheme."
                )
        self.weight_block_size = weight_block_size
        self.smooth = smooth
        self.mf_format = mf_format
        self.w_sparsity = w_sparsity
        self.mask_in_id = mask_in_id

    @classmethod
    def get_name(cls) -> str:
        return "blockwise_int8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BlockInt8Config:
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_int8_serialized = "int8" in quant_method
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        smooth = cls.get_from_keys_or(config, ["smooth"], False)
        mf_format = cls.get_from_keys_or(config, ["mf_format"], False)
        w_sparsity = cls.get_from_keys_or(config, ["w_sparsity"], 0.0)
        mask_in_id = cls.get_from_keys_or(config, ["mask_in_id"], False)
        return cls(
            is_checkpoint_int8_serialized=is_checkpoint_int8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
            smooth=smooth,
            mf_format=mf_format,
            w_sparsity=w_sparsity,
            mask_in_id=mask_in_id
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return BlockInt8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return BlockInt8MoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class BlockInt8LinearMethod(LinearMethodBase):
    """Linear method for INT8.
    Supports loading INT8 checkpoints with static weight scale and
    dynamic activation scale.

    Limitations:
    Only support block-wise int8 quantization and int8 checkpoint

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: BlockInt8Config):
        self.quant_config = quant_config
        assert self.quant_config.weight_block_size is not None
        assert self.quant_config.is_checkpoint_int8_serialized

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        tp_size = get_tensor_model_parallel_world_size()

        block_n, block_k = (
            self.quant_config.weight_block_size[0],
            self.quant_config.weight_block_size[1],
        )
        # Required by row parallel
        if tp_size > 1 and input_size // input_size_per_partition == tp_size:
            if input_size_per_partition % block_k != 0:
                raise ValueError(
                    f"Weight input_size_per_partition = "
                    f"{input_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}."
                )
        # Required by column parallel or enabling merged weights
        if (tp_size > 1 and output_size // output_size_per_partition == tp_size) or len(
            output_partition_sizes
        ) > 1:
            for output_partition_size in output_partition_sizes:
                if output_partition_size % block_n != 0:
                    raise ValueError(
                        f"Weight output_partition_size = "
                        f"{output_partition_size} is not divisible by "
                        f"weight quantization block_n = {block_n}."
                    )

        layer.logical_widths = output_partition_sizes

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # WEIGHT
        weight_dtype = (
            torch.int8
            if self.quant_config.is_checkpoint_int8_serialized
            else params_dtype
        )
        scale_dtype = (
            torch.bfloat16
            if self.quant_config.mf_format
            else torch.float32
        )

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=weight_dtype
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        scale = BlockQuantScaleParameter(
            data=torch.empty(
                (output_size_per_partition + block_n - 1) // block_n,
                (input_size_per_partition + block_k - 1) // block_k,
                dtype=scale_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        scale[:] = torch.finfo(scale_dtype).min
        layer.register_parameter("weight_scale_inv", scale)

        # low bits part
        if self.quant_config.w_sparsity > 0:
            # WEIGHT SCALE
            lscale = BlockQuantScaleParameter(
                data=torch.empty(
                    (output_size_per_partition + block_n - 1) // block_n,
                    (input_size_per_partition + block_k - 1) // block_k,
                    dtype=scale_dtype,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            lscale[:] = torch.finfo(scale_dtype).min
            layer.register_parameter("weight_lscale_inv", lscale)
            # MASK
            if self.quant_config.mask_in_id:
                # mask_id = _ColumnvLLMParameter(
                #     data=torch.empty(
                #         output_size_per_partition,
                #         (input_size_per_partition + block_k - 1) // block_k,
                #         int(
                #             self.quant_config.weight_block_size[1] * \
                #                 (1 - self.quant_config.w_sparsity)
                #         ),
                #         dtype=torch.int8
                #     ),
                #     output_dim=0,
                #     weight_loader=weight_loader
                # )
                mask_id = BlockQuantScaleParameter(
                    data=torch.empty(
                        (output_size_per_partition + block_n - 1) // block_n,
                        (input_size_per_partition + block_k - 1) // block_k,
                        int(
                            self.quant_config.weight_block_size[1] * \
                                self.quant_config.weight_block_size[0] * \
                                (1 - self.quant_config.w_sparsity)
                        ),
                        dtype=torch.int8
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=weight_loader
                )
                layer.register_parameter("mask_id", mask_id)
            else:
                mask = ModelWeightParameter(
                    data=torch.empty(
                        output_size_per_partition, input_size_per_partition, dtype=weight_dtype
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=weight_loader,
                )
                layer.register_parameter("mask", mask)
        else:
            layer.weight_lscale_inv = None
            layer.mask = 1

        # INPUT ACTIVATION SCALE
        assert self.quant_config.activation_scheme == "dynamic"
        layer.register_parameter("input_scale", None)

        # SMOOTH SCALE
        if self.quant_config.smooth:
            smooth_scale = RowvLLMParameter(
                data=torch.empty(
                    1,
                    input_size_per_partition,
                    dtype=torch.bfloat16,
                ),
                input_dim=1,
                weight_loader=weight_loader,
            )
            smooth_scale[:] = torch.finfo(torch.bfloat16).min
            layer.register_parameter("smooth_scale", smooth_scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        # Block quant doesn't need to process weights after loading
        # Use torch Parameter to avoid cuda graph capturing issue
        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(
            layer.weight_scale_inv.data, requires_grad=False
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.quant_config.smooth:
            x = x / layer.smooth_scale

        if self.quant_config.w_sparsity > 0 and self.quant_config.mask_in_id:
            mask = generate_mask(
                layer.mask_id,
                self.quant_config.weight_block_size,
                dtype=layer.weight.dtype,
            )
        else:
            mask = layer.mask
        
        output = apply_w8a8_block_int8_linear(
            input=x,
            weight=layer.weight * mask,
            block_size=self.quant_config.weight_block_size,
            weight_scale=layer.weight_scale_inv,
            input_scale=None,
            bias=bias,
            mf_format=self.quant_config.mf_format
        )

        if self.quant_config.w_sparsity > 0:
            output += apply_w8a8_block_int8_linear(
                input=x,
                weight=layer.weight * (1 - mask),
                block_size=self.quant_config.weight_block_size,
                weight_scale=layer.weight_lscale_inv,
                input_scale=None,
                bias=bias,
                mf_format=self.quant_config.mf_format
            )
        return output

class BlockInt8MoEMethod(FusedMoEMethodBase):
    """MoE method for INT8.
    Supports loading INT8 checkpoints with static weight scale and
    dynamic activation scale.

    Limitations:
    Only support block-wise int8 quantization and int8 checkpoint

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: BlockInt8Config):
        self.quant_config = quant_config
        assert self.quant_config.weight_block_size is not None
        assert self.quant_config.is_checkpoint_int8_serialized

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        if self.quant_config.is_checkpoint_int8_serialized:
            params_dtype = torch.int8
        scale_dtype = (
            torch.bfloat16
            if self.quant_config.mf_format
            else torch.float32
        )
        tp_size = get_tensor_model_parallel_world_size()

        block_n, block_k = (
            self.quant_config.weight_block_size[0],
            self.quant_config.weight_block_size[1],
        )
        # NOTE(HandH1998): To ensure proper alignment of the block-wise quantization scales, the output_size of the weights for both the gate and up layers must be divisible by block_n.
        # Required by column parallel or enabling merged weights
        if intermediate_size_per_partition % block_n != 0:
            raise ValueError(
                f"The output_size of gate's and up's weight = "
                f"{intermediate_size_per_partition} is not divisible by "
                f"weight quantization block_n = {block_n}."
            )
        if tp_size > 1:
            # Required by row parallel
            if intermediate_size_per_partition % block_k != 0:
                raise ValueError(
                    f"The input_size of down's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}."
                )

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        
        # low bits part
        if self.quant_config.w_sparsity > 0:
            # MASK
            if self.quant_config.mask_in_id:
                w13_mask_id = torch.nn.Parameter(
                    torch.empty(
                        num_experts,
                        2 * ((intermediate_size_per_partition + block_n - 1) // block_n),
                        (hidden_size + block_k - 1) // block_k,
                        int(
                            self.quant_config.weight_block_size[1] * \
                                self.quant_config.weight_block_size[0] * \
                                (1 - self.quant_config.w_sparsity)
                        ),
                        dtype=torch.int8,
                    ),
                    requires_grad=False
                )
                layer.register_parameter("w13_mask_id", w13_mask_id)
                set_weight_attrs(w13_mask_id, extra_weight_attrs)
                w2_mask_id = torch.nn.Parameter(
                    torch.empty(
                        num_experts,
                        (hidden_size + block_n - 1) // block_n,
                        (intermediate_size_per_partition + block_k - 1) // block_k,
                        int(
                            self.quant_config.weight_block_size[1] * \
                                self.quant_config.weight_block_size[0] * \
                                (1 - self.quant_config.w_sparsity)
                        ),
                        dtype=torch.int8,
                    ),
                    requires_grad=False
                )
                layer.register_parameter("w2_mask_id", w2_mask_id)
                set_weight_attrs(w2_mask_id, extra_weight_attrs)
            else:
                w13_mask = torch.nn.Parameter(
                    torch.empty(
                        num_experts,
                        2 * intermediate_size_per_partition,
                        hidden_size,
                        dtype=params_dtype,
                    ),
                    requires_grad=False,
                )
                layer.register_parameter("w13_mask", w13_mask)
                set_weight_attrs(w13_mask, extra_weight_attrs)

                w2_mask = torch.nn.Parameter(
                    torch.empty(
                        num_experts,
                        hidden_size,
                        intermediate_size_per_partition,
                        dtype=params_dtype,
                    ),
                    requires_grad=False,
                )
                layer.register_parameter("w2_mask", w2_mask)
                set_weight_attrs(w2_mask, extra_weight_attrs)
        else:
            layer.w13_mask = None
            layer.w2_mask = None

        # WEIGHT_SCALES
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * ((intermediate_size_per_partition + block_n - 1) // block_n),
                (hidden_size + block_k - 1) // block_k,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                (hidden_size + block_n - 1) // block_n,
                (intermediate_size_per_partition + block_k - 1) // block_k,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # low bits part
        if self.quant_config.w_sparsity > 0:
            # WEIGHT_SCALES
            w13_weight_lscale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * ((intermediate_size_per_partition + block_n - 1) // block_n),
                    (hidden_size + block_k - 1) // block_k,
                    dtype=scale_dtype,
                ),
                requires_grad=False,
            )
            w2_weight_lscale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size_per_partition + block_k - 1) // block_k,
                    dtype=scale_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_lscale_inv", w13_weight_lscale)
            layer.register_parameter("w2_weight_lscale_inv", w2_weight_lscale)
            set_weight_attrs(w13_weight_lscale, extra_weight_attrs)
            set_weight_attrs(w2_weight_lscale, extra_weight_attrs)
        else:
            layer.w13_weight_lscale_inv = None
            layer.w2_weight_lscale_inv = None

        # INPUT_SCALES
        assert self.quant_config.activation_scheme == "dynamic"
        layer.w13_input_scale = None
        layer.w2_input_scale = None

        # SMOOTH SCALE
        # TODO: to be fixed
        # if self.quant_config.smooth:
        if False:
            w13_smooth_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    1,
                    hidden_size,
                    dtype=scale_dtype,
                ),
                requires_grad=False,
            )
            w2_smooth_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    1,
                    intermediate_size_per_partition,
                    dtype=scale_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_smooth_scale", w13_smooth_scale)
            layer.register_parameter("w2_smooth_scale", w2_smooth_scale)
            set_weight_attrs(w13_smooth_scale, extra_weight_attrs)
            set_weight_attrs(w2_smooth_scale, extra_weight_attrs)
        else:
            layer.w13_smooth_scale = None
            layer.w2_smooth_scale = None

    def process_weights_after_loading(self, layer: Module) -> None:
        # Block quant doesn't need to process weights after loading
        return

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        if self.quant_config.w_sparsity > 0 and self.quant_config.mask_in_id:
            w13_mask = generate_mask(
                rearrange(layer.w13_mask_id, "E O (G I) N -> (E G O) I N", G=2),
                self.quant_config.weight_block_size,
                dtype=layer.w13_weight.dtype,
            )
            w13_mask = rearrange(
                w13_mask,
                "(E G O) I -> E O (G I)",
                E=layer.w13_weight.size(0),
                G=2
            )
            w2_mask = generate_mask(
                rearrange(layer.w2_mask_id, "E O I N -> (E O) I N"),
                self.quant_config.weight_block_size,
                dtype=layer.w2_weight.dtype,
            )
            w2_mask = rearrange(
                w2_mask, "(E O) I -> E O I", E=layer.w2_weight.size(0)
            )
        else:
            w13_mask = layer.w13_mask
            w2_mask = layer.w2_mask
        
        quant_info = TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            use_int8_w8a8=True,
            w13_mask=w13_mask, # moffett
            w2_mask=w2_mask, # moffett
            w13_scale=layer.w13_weight_scale_inv,
            w2_scale=layer.w2_weight_scale_inv,
            w13_lscale=layer.w13_weight_lscale_inv, # moffett
            w2_lscale=layer.w2_weight_lscale_inv, # moffett
            mf_format=self.quant_config.mf_format, # moffett
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            a13_smooth_scale=layer.w13_smooth_scale, # moffett: unimplemented
            a2_smooth_scale=layer.w2_smooth_scale, # moffett: unimplemented
            block_shape=self.quant_config.weight_block_size,
        )

        output = self.runner.run(dispatch_output, quant_info)
        
        return output
