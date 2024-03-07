""" Quantization techniques for FP8 """
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

from tvm import DataType, DataTypeCode, IRModule
from tvm import dlight as dl
from tvm import relax, te, tir, topi
from tvm.relax.frontend import nn
from tvm.runtime import NDArray
from tvm.target import Target

from mlc_chat.loader import QuantizeMapping
from mlc_chat.nn import MixtralExperts
from mlc_chat.support import logging
from mlc_chat.support import tensor_parallel as tp

from . import group_quantization as gq


def quantize(
    x: nn.Tensor, quantize_dtype: str, kind="fp8-max", name: str = "quantize", **kwargs
) -> Tuple[nn.Tensor, ...]:
    """
    Quantizes the input tensor to a specified lower-precision datatype using different quantization schemes.

    This function supports quantization schemes such as 'fp8-max', where each element in the tensor is scaled and
    quantized to a target datatype that uses fewer bits than the original datatype. The fp8-max range scheme
    scales the tensor values based on the maximum value in the tensor to utilize the full range of the target datatype.

    Parameters
    ----------
    x : nn.Tensor
        The input tensor to be quantized.

    quantize_dtype : DataType
        The target datatype for quantization.

    kind : str, optional
        The kind of quantization scheme to use.

    name : str, optional
        A name hint for the operation.

    **kwargs : dict
        Additional keyword arguments for quantization parameters. For 'fp8-max', 'max_int_value' must be provided,
        which defines the maximum integer value that can be represented in the target datatype.

    Returns
    -------
    result : Tuple[nn.Tensor, ...]
        A list of tensors from the qunatization,
        Usually the quantized tensor, and parameter tensors like scale and zero point

    """
    if kind == "fp8-max":
        # quant: Tensor(dtype="e4m3_float8") = (x / scale); scale: float16 = max(x) / fp8_max_int_value);
        assert (
            "max_int_value" in kwargs
        ), "'max_int_value' must be provided when using fp8-max quantization"

        def fused_compute_scale_and_quantize(
            tensor: te.Tensor, axis: int, out_shape: Optional[List[tir.PrimExpr]] = None
        ):
            max_int = tir.const(kwargs["max_int_value"], x.dtype)
            min_scaling_factor = tir.const(1.0 / (kwargs["max_int_value"] * 512.0), x.dtype)
            r_idx = [te.reduce_axis((0, d)) for d in tensor.shape]
            max_abs = te.compute(
                shape=(1,),
                fcompute=lambda *idx: te.max(
                    te.abs(tensor(*r_idx)),
                    axis=r_idx,
                ),
                name="max_abs_value",
            )
            scale = te.compute(
                (1,),
                lambda *idx: te.max(
                    max_abs(*idx).astype(x.dtype) / max_int,
                    min_scaling_factor,
                ),
                name="scale",
            )
            scaled_act = te.compute(
                shape=tensor.shape,
                fcompute=lambda *idx: tir.Cast(
                    quantize_dtype,
                    tensor(*idx) / scale[0],
                ),
            )

            return scaled_act, scale

        quant, scale = nn.op.tensor_expr_op(  # pylint: disable=invalid-name
            lambda tensor: fused_compute_scale_and_quantize(  # pylint: disable=protected-access
                tensor,
                axis=None,
                out_shape=x.shape,
            ),
            name_hint="quantize_act",
            args=[x],
        )
        return quant, scale
    else:
        raise ValueError("Unknown quantization kind")


def dequantize(
    quant: nn.Tensor,
    scale: nn.Tensor,
    zero: nn.Tensor = None,
    kind="fp8-max",
    name="dequantize",
    **kwargs,
) -> nn.Tensor:
    """
    Dequantizes the input tensor from a specified lower-precision datatype back to a higher-precision datatype.

    This function supports dequantization schemes such as 'fp8-max', where each element in the quantized tensor
    is converted back to a higher-precision format using the provided scale. The 'fp8-max' scheme specifically
    reverses the scaling applied during quantization, without utilizing a zero-point adjustment.

    Parameters
    ----------
    quant : nn.Tensor
        The quantized tensor to be dequantized.

    scale : nn.Tensor
        The scale used during quantization.
        original higher-precision format.

    zero : nn.Tensor, optional
        The zero-point used during quantization.

    kind : str, optional
        The kind of dequantization scheme to use.

    name : str, optional
        A name hint for the operation.

    **kwargs : dict
        Additional keyword arguments for dequantization parameters.

    Returns
    -------
    nn.Tensor
        The dequantized tensor.

    """
    if kind == "fp8-max":
        # dequant: Tensor(dtype="float16") = (quant * scale); scale precompute by quantization
        assert zero == None, "FP8 max range quantization does not utilzie a zero point"
        return quant * scale
    else:
        raise ValueError("Unknown quantization kind")


nn.op.quantize = quantize
nn.op.dequantize = dequantize


class GroupQuantizeLinearFP8E4M3CutlassScaleOnly(
    gq.GroupQuantizeLinear,
):  # pylint: disable=too-many-instance-attributes
    """An nn.Linear module with group quantization"""

    def forward(self, x: nn.Tensor) -> nn.Tensor:  # pylint: disable=invalid-name
        """
        Forward method for group quantized linear layer.

        Parameters
        ----------
        x : nn.Tensor
            The input tensor.

        Returns
        -------
        ret : nn.Tensor
            The output tensor for the group quantized linear layer.
        """
        assert self.config.fp8_quant
        assert DataType(self.config.quantize_dtype).type_code == DataTypeCode.E4M3Float
        # For cutlass mixed-dtype gemm activation layout is row major w/ shape (M, K)
        # and weight layout column major w/ shape (N, K) so no permute dims is needed
        assert self.config.linear_weight_layout == "NK"

        # TODO(csullivan): Add a workspace for static allocation and planning
        # tmp_out = op.wrap_nested(
        #     relax.op.builtin.alloc_tensor(
        #         relax.ShapeExpr(
        #             (num_tokens, num_q_heads, self.max_num_partitions, head_dim)
        #         ),
        #         dtype=query._expr.struct_info.dtype,
        #         runtime_device_index=0,
        #     ),
        #     "relax.alloc_tensor",
        # )

        M, K = x.shape
        N, _ = self.q_weight.shape
        if self.bias:
            return nn.op.extern(
                "cutlass.mixed_dtype_gemm_fp16_fp8_scale",
                [
                    x,
                    self.q_weight,
                    self.bias,
                    self.q_scale,
                    M,
                    N,
                    K,
                    1,
                    self.config.group_size
                    # tmp_out,
                ],
                x,
            )
        else:
            return nn.op.extern(
                "cutlass.mixed_dtype_matmul_fp16_fp8_scale",
                [
                    x,
                    self.q_weight,
                    self.q_scale,
                    M,
                    N,
                    K,
                    1,
                    self.config.group_size
                    # tmp_out,
                ],
                out=nn.Tensor.placeholder(
                    (M, N), dtype=self.out_dtype if self.out_dtype else self.config.model_dtype
                ),
            )


class MixtralExpertsFP8(
    gq.GroupQuantizeMixtralExperts
):  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        num_local_experts,
        in_features,
        out_features,
        weight_config: gq.GroupQuantize,
        activation_dtype: str = None,
        weight_dtype: str = None,
        runtime: str = "cast",
    ):  # pylint: disable=too-many-arguments
        super().__init__(num_local_experts, in_features, out_features, weight_config)
        self.activation_dtype = activation_dtype
        self.weight_dtype = weight_dtype
        self.runtime = runtime
        self.weight_config = weight_config
        self.max_int_value = 448 if "e4m3" in activation_dtype else 57344

        # TODO(csullivan): Need param allocation before this will work
        # if "max" in self.runtime:
        #     self.q_scale_activation = nn.Parameter((1,), weight_config.model_dtype)

    @staticmethod
    def from_mixtral_experts(
        src: "MixtralExperts",
        weight_config: gq.GroupQuantize,
        activation_dtype: str = None,
        weight_dtype: str = None,
        runtime: str = "cast",
    ) -> "GroupQuantizeMixtralExperts":
        """
        Converts a non-quantized MixtralExperts to a group quantized GroupQuantizeMixtralExperts

        Parameters
        ----------
        src : MixtralExperts
            The non-quantized MixtralExperts

        weight_config : GroupQuantize
            The group quantization weight_config.

        Returns
        -------
        ret : GroupQuantizeMixtralExperts
            The group quantized GroupQuantizeMixtralExperts layer.
        """
        quantized_mistral_experts = MixtralExpertsFP8(
            num_local_experts=src.num_local_experts,
            in_features=src.in_features,
            out_features=src.out_features,
            weight_config=weight_config,
            activation_dtype=activation_dtype,
            weight_dtype=weight_dtype,
            runtime=runtime,
        )
        if "shard_strategy" in src.weight.attrs:
            shard = src.weight.attrs["shard_strategy"]
            gq._apply_sharding(shard, f"{shard.name}_q_weight", quantized_mistral_experts.q_weight)
            if weight_dtype == "e4m3_float8":
                _apply_sharding(shard, f"{shard.name}_q_scale", quantized_mistral_experts.q_scale)
            # TODO(csullivan): Consider if it is possible to use all reduce to reduce over multigpu
            # calibration params. Probably doable with nn.op.ccl* directly

        return quantized_mistral_experts

    def forward(self, x: nn.Tensor, indptr: nn.Tensor) -> nn.Tensor:  # pylint: disable=invalid-name
        if self.runtime == "max-calibration":
            x, scale = nn.op.quantize(
                x,
                quantize_dtype=self.activation_dtype,
                kind="fp8-max",
                max_int_value=self.max_int_value,
            )
            # TODO(csullivan): support weight update
            # self.q_scale_activation = nn.op.maximum(self.q_scale_activation, scale)
        elif self.runtime == "max":
            x = x / self.q_scale_activation
            x = nn.op.astype(x, dtype=self.activation_dtype)
        elif self.runtime == "cast":
            x = nn.op.astype(x, dtype=self.activation_dtype)
        else:
            raise NotImplementedError(
                f"Only max and cast runtimes are supported for FP8 activations, {self.runtime} is unsupported."
            )

        workspace = nn.op.wrap_nested(
            relax.op.builtin.alloc_tensor(
                relax.ShapeExpr((4096 * 1024,)),
                dtype="uint8",
                runtime_device_index=0,
            ),
            "relax.alloc_tensor",
        )

        batch_size, in_features = x.shape
        num_local_experts, out_features, _ = self.q_weight.shape
        a_format = self.activation_dtype.split("_")[0]
        w_format = self.weight_dtype.split("_")[0]

        output = nn.op.extern(
            f"cutlass.moe_gemm_{a_format}_{w_format}_fp16",
            [
                x,
                self.q_weight,
                indptr,
                workspace,
                batch_size,
                out_features,
                in_features,
                num_local_experts,
            ],
            out=nn.Tensor.placeholder(
                (batch_size, out_features),
                dtype=self.weight_config.model_dtype,
            ),
        )

        if self.runtime == "cast":
            return output

        # TODO(csullivan): absorb these scales into the group gemm
        if self.weight_dtype == "e4m3_float8":
            total_scale = scale * self.q_scale
        else:
            total_scale = scale

        return nn.op.dequantize(output, total_scale)