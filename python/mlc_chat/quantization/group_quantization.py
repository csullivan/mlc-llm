"""The group quantization config"""

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


from .utils import convert_uint_to_float, is_final_fc, convert_uint_packed_fp8_to_float

logger = logging.getLogger(__name__)


@dataclass
class GroupQuantize:  # pylint: disable=too-many-instance-attributes
    """Configuration for group quantization"""

    name: str
    kind: str
    group_size: int
    quantize_dtype: Literal["int3", "int4", "int8", "e4m3_float8", "e5m2_float8"]
    storage_dtype: Literal["uint32"]
    model_dtype: Literal["float16", "float32"]
    linear_weight_layout: Literal["KN", "NK"]
    quantize_embedding: bool = True
    quantize_final_fc: bool = True

    num_elem_per_storage: int = 0
    num_storage_per_group: int = 0
    max_int_value: int = 0

    def __post_init__(self):
        assert self.kind == "group-quant"
        quantize_dtype = DataType(self.quantize_dtype)
        storage_dtype = DataType(self.storage_dtype)
        model_dtype = DataType(self.model_dtype)
        if quantize_dtype.type_code in (DataTypeCode.E4M3Float, DataTypeCode.E5M2Float):
            self.fp8_quant = True
        else:
            self.fp8_quant = False
            assert quantize_dtype.type_code == DataTypeCode.INT
        self.no_scale = quantize_dtype.type_code == DataTypeCode.E5M2Float
        assert storage_dtype.type_code == DataTypeCode.UINT
        assert model_dtype.type_code == DataTypeCode.FLOAT
        if storage_dtype.bits < quantize_dtype.bits:
            raise ValueError("Storage unit should be greater or equal to quantized element")

        self.num_elem_per_storage = storage_dtype.bits // quantize_dtype.bits
        if self.group_size % self.num_elem_per_storage != 0:
            raise ValueError("Group size should be divisible by numbers of elements per storage")
        self.num_storage_per_group = self.group_size // self.num_elem_per_storage
        if self.fp8_quant:
            if quantize_dtype.type_code == DataTypeCode.E4M3Float:
                self.max_int_value = 448
            elif quantize_dtype.type_code == DataTypeCode.E5M2Float:
                self.max_int_value = 57344
            else:
                raise NotImplementedError()
        else:
            self.max_int_value = (2 ** (quantize_dtype.bits - 1)) - 1
        self.linear_quant_axis = 0 if self.linear_weight_layout == "KN" else 1
        self._quantize_func_cache = {}

    def quantize_model(
        self,
        model: nn.Module,
        quant_map: QuantizeMapping,
        name_prefix: str,
    ) -> nn.Module:
        """
        Quantize model with group quantization

        Parameters
        ----------
        model : nn.Module
            The non-quantized nn.Module.

        quant_map : QuantizeMapping
            The quantize mapping with name mapping and func mapping.

        name_prefix : str
            The name prefix for visited weight.

        Returns
        -------
        ret : nn.Module
            The quantized nn.Module.
        """

        class _Mutator(nn.Mutator):
            def __init__(self, config: GroupQuantize, quant_map: QuantizeMapping) -> None:
                super().__init__()
                self.config = config
                self.quant_map = quant_map

            def visit_module(self, name: str, node: nn.Module) -> Any:
                """
                The visiting method for group quantization of nn.Module nodes.

                Parameters
                ----------
                name : str
                    The name of the current node.

                node : nn.Module
                    The current node of nn.Module to mutate.

                Returns
                ------
                ret_node: Any
                    The new node to replace current node.
                """
                if isinstance(node, nn.Linear) and (
                    not is_final_fc(name) or self.config.quantize_final_fc
                ):
                    weight_name = f"{name}.weight"
                    self.quant_map.param_map[weight_name] = [f"{name}.q_weight", f"{name}.q_scale"] if self.config.no_scale else [f"{name}.q_weight",]
                    self.quant_map.map_func[weight_name] = partial(
                        self.config.quantize_weight,
                        output_transpose=self.config.linear_weight_layout == "KN",
                    )
                    if False and self.config.quantize_dtype == "e4m3_float8":
                        return GroupQuantizeLinearFP8E4M3ScaleOnly.from_linear(node, self.config)
                    elif False and self.config.quantize_dtype == "e5m2_float8":
                        return GroupQuantizeLinearFP8E5M2.from_linear(node, self.config)
                    else:
                        return GroupQuantizeLinear.from_linear(node, self.config)
                if isinstance(node, nn.Embedding) and self.config.quantize_embedding:
                    weight_name = f"{name}.weight"
                    self.quant_map.param_map[weight_name] = [f"{name}.q_weight", f"{name}.q_scale"] if self.config.no_scale else [f"{name}.q_weight",]
                    self.quant_map.map_func[weight_name] = self.config.quantize_weight
                    return GroupQuantizeEmbedding.from_embedding(node, self.config)
                if isinstance(node, MixtralExperts):
                    weight_name = f"{name}.weight"
                    self.quant_map.param_map[weight_name] = [f"{name}.q_weight", f"{name}.q_scale"] if self.config.no_scale else [f"{name}.q_weight",]
                    self.quant_map.map_func[weight_name] = self.config.quantize_weight
                    return GroupQuantizeMixtralExperts.from_mixtral_experts(node, self.config)
                return self.visit(name, node)

        model.to(dtype=self.model_dtype)
        mutator = _Mutator(self, quant_map)
        model = mutator.visit(name_prefix, model)
        return model

    def _dequantize(
        self,
        weight: te.Tensor,
        scale: te.Tensor,
        axis: int,
        out_shape: Optional[List[tir.PrimExpr]] = None,
    ):
        tir_max_int = tir.const(self.max_int_value, self.model_dtype)
        float_weight = convert_uint_to_float(
            weight,
            DataType(self.quantize_dtype).bits,
            self.num_elem_per_storage,
            self.storage_dtype,
            self.model_dtype,
            axis=axis,
            out_shape=out_shape,
        )
        if out_shape is None:
            out_shape = weight.shape
            out_shape[axis] *= self.num_elem_per_storage
        axis = axis if axis >= 0 else len(out_shape) + axis
        return te.compute(
            shape=out_shape,
            fcompute=lambda *idx: tir.multiply(
                tir.subtract(
                    float_weight(*idx),
                    tir_max_int,  # TODO(jmcmahan): the max_int shift to remove negatives is not necessary for fp8
                ),
                scale(*idx[:axis], idx[axis] // self.group_size, *idx[axis + 1 :]),
            ),
            name="dequantize",
        )

    def _dequantize_e4m3(
        self,
        weight: te.Tensor,
        scale: te.Tensor,
        axis: int,
        out_shape: Optional[List[tir.PrimExpr]] = None,
    ):
        float_e4m3_weight = convert_uint_packed_fp8_to_float(
            weight,
            DataType(self.quantize_dtype).bits,
            self.num_elem_per_storage,
            self.storage_dtype,
            self.model_dtype,
            self.quantize_dtype,
            axis=axis,
            out_shape=out_shape,
        )
        if out_shape is None:
            out_shape = weight.shape
            out_shape[axis] *= self.num_elem_per_storage
        axis = axis if axis >= 0 else len(out_shape) + axis
        return te.compute(
            shape=out_shape,
            fcompute=lambda *idx: tir.multiply(
                float_e4m3_weight(*idx),
                scale(*idx[:axis], idx[axis] // self.group_size, *idx[axis + 1 :]),
            ),
            name="dequantize",
        )

    def _dequantize_e5m2(
        self,
        weight: te.Tensor,
        axis: int,  # axis is still relevant, because it determines which axis is packed into u32 storage
        out_shape: Optional[List[tir.PrimExpr]] = None,
    ):
        float_e5m2_weight = convert_uint_packed_fp8_to_float(
            weight,
            DataType(self.quantize_dtype).bits,
            self.num_elem_per_storage,
            self.storage_dtype,
            self.model_dtype,
            self.quantize_dtype,
            axis=axis,
            out_shape=out_shape,
        )
        return float_e5m2_weight

    def quantize_weight(
        self, weight: NDArray, axis: int = -1, output_transpose: bool = False
    ) -> List[NDArray]:
        """
        Quantize weight with group quantization

        Parameters
        ----------
        weight : NDArray
            The original weight.

        axis : int
            The group axis.

        output_transpose : bool
            Whether to transpose the output quantized weight. Only 2D weight is supported.

        Returns
        ------
        ret: List[NDArray]
            The list of group quantized weights.
        """
        device = weight.device
        device_type = device.MASK2STR[device.device_type]
        axis = axis if axis >= 0 else len(weight.shape) + axis

        def _create_quantize_func() -> IRModule:
            if self.fp8_quant:
                if (
                    DataType(self.quantize_dtype).type_code == DataTypeCode.E4M3Float
                    or DataType(self.quantize_dtype).type_code == DataTypeCode.E5M2Float
                ):
                    quantize_func = self._quantize_float8
                else:
                    assert NotImplementedError()
            else:
                quantize_func = self._quantize

            bb = relax.BlockBuilder()  # pylint: disable=invalid-name
            weight_var = relax.Var("weight", relax.TensorStructInfo(weight.shape, weight.dtype))
            with bb.function(name="main", params=[weight_var]):
                with bb.dataflow():
                    lv = bb.emit_te(quantize_func, weight_var, axis, output_transpose)
                    gv = bb.emit_output(lv)  # pylint: disable=invalid-name
                bb.emit_func_output(gv)
            return bb.finalize()

        def _compile_quantize_func(mod: IRModule) -> Callable:
            if device_type in ["cuda", "rocm", "metal", "vulkan"]:
                target = Target.current()
                if target is None:
                    target = Target.from_device(device)
                with target:
                    mod = dl.ApplyDefaultSchedule(  # type: ignore   # pylint: disable=not-callable
                        dl.gpu.Reduction(),
                        dl.gpu.GeneralReduction(),
                        dl.gpu.Fallback(),
                    )(mod)
            elif device_type == "cpu":
                target = "llvm"
                mod = relax.transform.LegalizeOps()(mod)
            else:
                raise NotImplementedError(f"Device type {device_type} is not supported")
            ex = relax.build(mod, target=target)
            vm = relax.VirtualMachine(ex, device)  # pylint: disable=invalid-name
            return vm["main"]

        key = (
            f"({weight.shape}, {weight.dtype}, {device_type}, "
            f"axis={axis}, output_transpose={output_transpose})"
        )
        quantize_func = self._quantize_func_cache.get(key, None)
        if quantize_func is None:
            logger.info("Compiling quantize function for key: %s", key)
            quantize_func = _compile_quantize_func(_create_quantize_func())
            self._quantize_func_cache[key] = quantize_func
        return quantize_func(weight)

    def _quantize(  # pylint: disable=too-many-locals
        self,
        weight: te.Tensor,
        axis: int = -1,
        output_transpose: bool = False,
    ) -> Tuple[te.Tensor, te.Tensor]:
        """Group quantization for weight tensor, defined in tensor expression."""
        max_int = tir.const(self.max_int_value, self.model_dtype)
        shape = weight.shape  # pylint: disable=invalid-name
        axis = axis if axis >= 0 else len(shape) + axis
        k = shape[axis]
        quantize_dtype = DataType(self.quantize_dtype)
        # compute scale per group
        r = te.reduce_axis((0, self.group_size), name="r")  # pylint: disable=invalid-name
        num_group = tir.ceildiv(k, self.group_size)
        scale_shape = (*shape[:axis], num_group, *shape[axis + 1 :])
        max_abs = te.compute(
            shape=scale_shape,
            fcompute=lambda *idx: te.max(
                tir.if_then_else(
                    idx[axis] * self.group_size + r < k,
                    te.abs(weight(*idx[:axis], idx[axis] * self.group_size + r, *idx[axis + 1 :])),
                    te.min_value(self.model_dtype),
                ),
                axis=r,
            ),
            name="max_abs_value",
        )
        scale = te.compute(
            scale_shape,
            lambda *idx: max_abs(*idx).astype(self.model_dtype) / max_int,
            name="scale",
        )
        # compute scaled weight
        scaled_weight = te.compute(
            shape=weight.shape,
            fcompute=lambda *idx: tir.min(
                tir.max(
                    tir.round(
                        weight(*idx)
                        / scale(*idx[:axis], idx[axis] // self.group_size, *idx[axis + 1 :])
                        + max_int
                    ),
                    tir.const(0, self.model_dtype),
                ),
                max_int * 2,
            ).astype(self.storage_dtype),
        )
        # compute quantized weight per storage
        r = te.reduce_axis((0, self.num_elem_per_storage), name="r")  # pylint: disable=invalid-name
        num_storage = self.num_storage_per_group * num_group
        quantized_weight_shape = (*shape[:axis], num_storage, *shape[axis + 1 :])
        quantized_weight = te.compute(
            shape=quantized_weight_shape,
            fcompute=lambda *idx: tir.sum(
                tir.if_then_else(
                    idx[axis] * self.num_elem_per_storage + r < k,
                    scaled_weight(
                        *idx[:axis], idx[axis] * self.num_elem_per_storage + r, *idx[axis + 1 :]
                    )
                    << (r * quantize_dtype.bits),
                    0,
                ),
                axis=r,
            ),
            name="weight",
        )
        if output_transpose:
            if len(quantized_weight.shape) != 2 or len(scale.shape) != 2:
                raise ValueError(
                    "Does not support transpose output quantized weight with ndim != 2"
                )
            quantized_weight = topi.transpose(quantized_weight)
            scale = topi.transpose(scale)
        return quantized_weight, scale

    def _quantize_float8(  # pylint: disable=too-many-locals
        self,
        weight: te.Tensor,
        axis: int = -1,
        output_transpose: bool = False,
    ) -> Tuple[te.Tensor, te.Tensor]:
        """Group quantization for weight tensor, defined in tensor expression."""

        shape = weight.shape  # pylint: disable=invalid-name
        quantize_dtype = DataType(self.quantize_dtype)
        k = shape[axis]

        if quantize_dtype.type_code == DataTypeCode.E4M3Float:
            # compute scale per group
            num_group = tir.ceildiv(k, self.group_size)
            max_int = tir.const(self.max_int_value, self.model_dtype)
            axis = axis if axis >= 0 else len(shape) + axis
            r = te.reduce_axis((0, self.group_size), name="r")  # pylint: disable=invalid-name
            scale_shape = (*shape[:axis], num_group, *shape[axis + 1 :])

            # min_scaling_factor taken from TRT-LLM
            min_scaling_factor = tir.const(1.0 / (self.max_int_value * 512.0), self.model_dtype)
            max_abs = te.compute(
                shape=scale_shape,
                fcompute=lambda *idx: te.max(
                    tir.if_then_else(
                        idx[axis] * self.group_size + r < k,
                        te.abs(
                            weight(*idx[:axis], idx[axis] * self.group_size + r, *idx[axis + 1 :])
                        ),
                        te.min_value(self.model_dtype),
                    ),
                    axis=r,
                ),
                name="max_abs_value",
            )
            scale = te.compute(
                scale_shape,
                lambda *idx: te.max(
                    max_abs(*idx).astype(self.model_dtype) / max_int, min_scaling_factor
                ),
                name="scale",
            )
            # compute scaled weight
            # TODO(fp8-team): Convince ourselves that we don't need to clip the quantized weight
            # Need a cast to FP8, then reinerpret cast
            scaled_weight = te.compute(
                shape=weight.shape,
                fcompute=lambda *idx: tir.reinterpret(
                    # TODO(csullivan) Change this to a vector type to simplify storage and improving casting
                    self.storage_dtype,
                    tir.Cast(
                        self.quantize_dtype,
                        weight(*idx)
                        / scale(*idx[:axis], idx[axis] // self.group_size, *idx[axis + 1 :]),
                    ),
                ),
            )
        elif quantize_dtype.type_code == DataTypeCode.E5M2Float:
            scaled_weight = te.compute(
                shape=weight.shape,
                fcompute=lambda *idx: tir.reinterpret(
                    self.storage_dtype,
                    tir.Cast(  # TODO(jmcmahan): verify that this cast (fp16 -> e5m2) does the expected mantissa clip
                        self.quantize_dtype, weight(*idx)
                    ),
                ),
            )

        # TODO(csullivan): If using vector type fp8x4 this compute op can be deleted
        # compute quantized weight per storage
        r = te.reduce_axis((0, self.num_elem_per_storage), name="r")  # pylint: disable=invalid-name
        num_storage = tir.ceildiv(k, self.num_elem_per_storage)
        quantized_weight_shape = (*shape[:axis], num_storage, *shape[axis + 1 :])
        quantized_weight = te.compute(
            shape=quantized_weight_shape,
            fcompute=lambda *idx: tir.sum(
                tir.if_then_else(
                    idx[axis] * self.num_elem_per_storage + r < k,
                    scaled_weight(
                        *idx[:axis], idx[axis] * self.num_elem_per_storage + r, *idx[axis + 1 :]
                    )
                    << (r * quantize_dtype.bits),
                    0,
                ),
                axis=r,
            ),
            name="weight",
        )

        if output_transpose:
            if len(quantized_weight.shape) != 2 or len(scale.shape) != 2:
                raise ValueError(
                    "Does not support transpose output quantized weight with ndim != 2"
                )
            quantized_weight = topi.transpose(quantized_weight)
            scale = topi.transpose(scale)
        if quantize_dtype.type_code == DataTypeCode.E4M3Float:
            return quantized_weight, scale
        elif quantize_dtype.type_code == DataTypeCode.E5M2Float:
            return quantized_weight


class GroupQuantizeLinear(nn.Module):  # pylint: disable=too-many-instance-attributes
    """An nn.Linear module with group quantization"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_features: int,
        out_features: Union[int, tir.Var],
        config: GroupQuantize,
        bias: bool = True,
        out_dtype: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.config = config
        self.no_scale = self.config.no_scale
        num_group = tir.ceildiv(in_features, config.group_size)
        if config.linear_weight_layout == "KN":
            self.q_weight = nn.Parameter(
                (config.num_storage_per_group * num_group, out_features), config.storage_dtype
            )
            if not self.no_scale:
                self.q_scale = nn.Parameter((num_group, out_features), config.model_dtype)
        else:
            self.q_weight = nn.Parameter(
                (out_features, config.num_storage_per_group * num_group), config.storage_dtype
            )
            if not self.no_scale:
                self.q_scale = nn.Parameter((out_features, num_group), config.model_dtype)
        if bias:
            self.bias = nn.Parameter(
                (out_features,), config.model_dtype if out_dtype is None else out_dtype
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, src: nn.Linear, config: GroupQuantize) -> "GroupQuantizeLinear":
        """
        Converts a non-quantized nn.Linear to a group quantized GroupQuantizeLinear

        Parameters
        ----------
        src : nn.Linear
            The non-quantized nn.Linear.

        config : GroupQuantize
            The group quantization config.

        Returns
        -------
        ret : GroupQuantizeLinear
            The group quantized GroupQuantizeLinear layer.
        """
        # For dynamic shape, src.out_features is `"name"`; src.weight.shape[0] is `tir.Var("name")`
        out_features, in_features = src.weight.shape
        quantized_linear = cls(
            in_features=in_features,
            out_features=out_features,
            config=config,
            bias=getattr(src, "bias", None) is not None,
            out_dtype=src.out_dtype,
        )
        if quantized_linear.bias is not None:
            quantized_linear.bias.attrs = src.bias.attrs
        if "shard_strategy" in src.weight.attrs:
            shard = src.weight.attrs["shard_strategy"]
            _apply_sharding(shard, f"{shard.name}_q_weight", quantized_linear.q_weight)
            if (
                not DataType(config.quantize_dtype).type_code == DataTypeCode.E5M2Float
            ):  # no scale for e5m2
                _apply_sharding(shard, f"{shard.name}_q_scale", quantized_linear.q_scale)
        return quantized_linear

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
        if self.config.fp8_quant:
            if DataType(self.config.quantize_dtype).type_code == DataTypeCode.E4M3Float:
                dequant_func = self.config._dequantize_e4m3
            elif DataType(self.config.quantize_dtype).type_code == DataTypeCode.E5M2Float:
                dequant_func = self.config._dequantize_e5m2
            else:
                raise NotImplementedError()
        else:
            dequant_func = self.confg._dequantize

        if self.config.linear_weight_layout == "NK":
            out_shape = [
                tir.IntImm("int64", self.out_features)
                if isinstance(self.out_features, int)
                else weight.shape[0],
                tir.IntImm("int64", self.in_features),
            ]
        else:
            out_shape = [
                tir.IntImm("int64", self.in_features),
                tir.IntImm("int64", self.out_features)
                if isinstance(self.out_features, int)
                else weight.shape[1],
            ]

        if not self.no_scale:
            w = nn.op.tensor_expr_op(  # pylint: disable=invalid-name
                lambda weight, scale: dequant_func(  # pylint: disable=protected-access
                    weight,
                    scale,
                    axis=self.config.linear_quant_axis,
                    out_shape=out_shape,
                ),
                name_hint="dequantize",
                args=[self.q_weight, self.q_scale],
            )
        else:
            w = nn.op.tensor_expr_op(  # pylint: disable=invalid-name
                lambda weight: dequant_func(  # pylint: disable=protected-access
                    weight,
                    axis=self.config.linear_quant_axis,
                    out_shape=out_shape,
                ),
                name_hint="dequantize",
                args=[
                    self.q_weight,
                ],
            )

        if self.config.linear_weight_layout == "NK":
            w = nn.op.permute_dims(w)  # pylint: disable=invalid-name
        x = nn.op.matmul(x, w, out_dtype=self.out_dtype)
        if self.bias is not None:
            x = x + self.bias
        return x

    def to(self, dtype: Optional[str] = None) -> None:
        """
        Override to() such that we do not convert bias if there is an out_dtype.
        Otherwise, we might run into dtype mismatch when computing x + self.bias.
        """
        self.q_weight.to(dtype=dtype)
        if not self.no_scale:
            self.q_scale.to(dtype=dtype)
        if self.bias is not None and self.out_dtype is None:
            self.bias.to(dtype=dtype)
        if dtype is not None and isinstance(getattr(self, "dtype", None), str):
            self.dtype = dtype  # pylint: disable=attribute-defined-outside-init


class GroupQuantizeEmbedding(nn.Module):
    """An nn.Embedding module with group quantization"""

    def __init__(self, num: Union[int, tir.Var], dim: int, config: GroupQuantize):
        self.num = num
        self.dim = dim
        self.config = config
        self.no_scale = self.config.no_scale
        num_group = tir.ceildiv(dim, config.group_size)
        self.q_weight = nn.Parameter(
            (num, config.num_storage_per_group * num_group), config.storage_dtype
        )
        if not self.no_scale:
            self.q_scale = nn.Parameter((num, num_group), config.model_dtype)

    @staticmethod
    def from_embedding(embedding: nn.Embedding, config: GroupQuantize) -> "GroupQuantizeEmbedding":
        """
        Converts a non-quantized nn.Embedding to a group quantized GroupQuantizeEmbedding

        Parameters
        ----------
        linear : nn.Embedding
            The non-quantized nn.Embedding.

        config : GroupQuantize
            The group quantization config.

        Returns
        -------
        ret : GroupQuantizeEmbedding
            The group quantized GroupQuantizeEmbedding layer.
        """
        num, dim = embedding.weight.shape
        return GroupQuantizeEmbedding(num, dim, config)

    def forward(self, x: nn.Tensor):  # pylint: disable=invalid-name
        """
        Forward method for group quantized embedding layer.

        Parameters
        ----------
        x : nn.Tensor
            The input tensor.

        Returns
        -------
        ret : nn.Tensor
            The output tensor for the embedding layer.
        """
        if self.config.fp8_quant:
            if DataType(self.config.quantize_dtype).type_code == DataTypeCode.E4M3Float:
                dequant_func = self.config._dequantize_e4m3
            elif DataType(self.config.quantize_dtype).type_code == DataTypeCode.E5M2Float:
                dequant_func = self.config._dequantize_e5m2
            else:
                raise NotImplementedError()
        else:
            dequant_func = self.confg._dequantize

        out_shape = [
            (
                tir.IntImm("int64", self.num) if isinstance(self.num, int) else weight.shape[0]
            ),  # Reuse same tir.Var for symbolic shape (after Exporter)
            tir.IntImm("int64", self.dim),
        ]

        if not self.no_scale:
            w = nn.op.tensor_expr_op(  # pylint: disable=invalid-name
                lambda weight, scale: dequant_func(  # pylint: disable=protected-access
                    weight,
                    scale,
                    axis=-1,
                    out_shape=out_shape,
                ),
                name_hint="dequantize",
                args=[self.q_weight, self.q_scale],
            )
        else:
            w = nn.op.tensor_expr_op(  # pylint: disable=invalid-name
                lambda weight: dequant_func(  # pylint: disable=protected-access
                    weight,
                    axis=-1,
                    out_shape=out_shape,
                ),
                name_hint="dequantize",
                args=[
                    self.q_weight,
                ],
            )

        if x.ndim == 1:
            return nn.op.take(w, x, axis=0)
        return nn.op.reshape(
            nn.op.take(w, nn.op.reshape(x, shape=[-1]), axis=0),
            shape=[*x.shape, self.dim],
        )

    def lm_head_forward(self, x: nn.Tensor):
        """The lm_head forwarding, which dequantizes the weight
        and multiplies it with the input tensor.

        Parameters
        ----------
        x : nn.Tensor
            The input tensor.

        Returns
        -------
        ret : nn.Tensor
            The output tensor for the lm_head layer.
        """
        if not self.no_scale:
            w = nn.op.tensor_expr_op(  # pylint: disable=invalid-name
                lambda weight, scale: self.config._dequantize(  # pylint: disable=protected-access
                    weight,
                    scale,
                    axis=-1,
                    out_shape=[
                        (
                            tir.IntImm("int64", self.num)
                        if isinstance(self.num, int)
                        else weight.shape[0]
                        ),
                        tir.IntImm("int64", self.dim),
                    ],
                ),
                name_hint="dequantize",
                args=[self.q_weight, self.q_scale],
            )
        else:
            w = nn.op.tensor_expr_op(  # pylint: disable=invalid-name
                lambda weight: self.config._dequantize(  # pylint: disable=protected-access
                    weight,
                    axis=-1,
                    out_shape=[
                        (
                            tir.IntImm("int64", self.num)
                        if isinstance(self.num, int)
                        else weight.shape[0]
                        ),
                        tir.IntImm("int64", self.dim),
                    ],
                ),
                name_hint="dequantize",
                args=[self.q_weight],
            )
        w = nn.op.permute_dims(w)
        return nn.op.matmul(x, w, out_dtype="float32")


class GroupQuantizeMixtralExperts(nn.Module):  # pylint: disable=too-many-instance-attributes
    """An MixtralExperts module with group quantization"""

    def __init__(
        self,
        num_local_experts,
        in_features,
        out_features,
        config: GroupQuantize,
    ):  # pylint: disable=too-many-arguments
        self.num_local_experts = num_local_experts
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.no_scale = self.config.no_scale
        num_group = tir.ceildiv(in_features, config.group_size)
        self.q_weight = nn.Parameter(
            (num_local_experts, out_features, config.num_storage_per_group * num_group),
            config.storage_dtype,
        )
        if not self.no_scale:
            self.q_scale = nn.Parameter(
                (num_local_experts, out_features, num_group), config.model_dtype
            )
        self.quantize_dtype = config.quantize_dtype
        self.group_size = config.group_size
        self.dtype = config.model_dtype
        if config.linear_weight_layout == "KN":
            raise NotImplementedError("GroupQuantizeMixtralExperts does not support KN layout now.")

    @staticmethod
    def from_mixtral_experts(
        src: "MixtralExperts", config: GroupQuantize
    ) -> "GroupQuantizeMixtralExperts":
        """
        Converts a non-quantized MixtralExperts to a group quantized GroupQuantizeMixtralExperts

        Parameters
        ----------
        src : MixtralExperts
            The non-quantized MixtralExperts

        config : GroupQuantize
            The group quantization config.

        Returns
        -------
        ret : GroupQuantizeMixtralExperts
            The group quantized GroupQuantizeMixtralExperts layer.
        """
        quantized_mistral_experts = GroupQuantizeMixtralExperts(
            num_local_experts=src.num_local_experts,
            in_features=src.in_features,
            out_features=src.out_features,
            config=config,
        )
        if "shard_strategy" in src.weight.attrs:
            shard = src.weight.attrs["shard_strategy"]
            _apply_sharding(shard, f"{shard.name}_q_weight", quantized_mistral_experts.q_weight)
            if not config.no_scale:
                _apply_sharding(shard, f"{shard.name}_q_scale", quantized_mistral_experts.q_scale)
        return quantized_mistral_experts

    def forward(self, x: nn.Tensor, indptr: nn.Tensor) -> nn.Tensor:  # pylint: disable=invalid-name
        """Forward method for group quantized mistral experts.

        Parameters
        ----------
        x : nn.Tensor
            The input tensor.

        indptr: nn.Tensor
            The indptr tensor

        single_batch_decode: bool
            Whether to use single-batch decode

        Returns
        -------
        ret : nn.Tensor
            The output tensor for the group quantized mistral experts layer.
        """
        from mlc_chat.op import moe_matmul  # pylint: disable=import-outside-toplevel

        assert x.ndim == 2
        if indptr.ndim == 2:  # single-batch
            assert indptr.shape[0] == 1
            if not self.no_scale:
                return moe_matmul.dequantize_gemv(
                    x,
                    self.q_weight,
                    self.q_scale,
                    indptr,
                    quantize_dtype=self.quantize_dtype,
                    group_size=self.group_size,
                )
            else:
                return moe_matmul.dequantize_gemv_no_scale(
                    x,
                    self.q_weight,
                    indptr,
                    quantize_dtype=self.quantize_dtype,
                    group_size=self.group_size,
                )
        assert indptr.ndim == 1
        if not self.no_scale:
            return moe_matmul.dequantize_group_gemm(
                x,
                self.q_weight,
                self.q_scale,
                indptr,
                quantize_dtype=self.quantize_dtype,
                indptr_dtype=indptr.dtype,
                group_size=self.group_size,
            )
        else:
            return moe_matmul.dequantize_group_gemm_no_scale(
                x,
                self.q_weight,
                indptr,
                quantize_dtype=self.quantize_dtype,
                indptr_dtype=indptr.dtype,
                group_size=self.group_size,
            )


def _apply_sharding(shard, name: str, weight: nn.Parameter):
    if isinstance(shard, tp.ShardSingleDim):
        weight.attrs["shard_strategy"] = tp.ShardSingleDim(
            name=name,
            dim=shard.dim,
            segs=shard.segs,
        )
    else:
        raise NotImplementedError(f"Unknowing sharding strategy: {shard}")


class GroupQuantizeLinearFP8E4M3ScaleOnly(
    GroupQuantizeLinear,
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


class GroupQuantizeLinearFP8E5M2(
    GroupQuantizeLinear,
):  # pylint: disable=too-many-instance-attributes
    """An nn.Linear module with group quantization"""

    def forward(self, x: nn.Tensor) -> nn.Tensor:  # pylint: disable=invalid-name
        pass
