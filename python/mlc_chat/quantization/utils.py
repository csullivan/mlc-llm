"""Common utilities for quantization"""
from typing import List, Optional

from tvm import te, tir
from tvm.script import tir as T


def convert_uint_to_float(  # pylint: disable=too-many-arguments
    weight: te.Tensor,
    bits: int,
    num_elem_per_storage: int,
    storage_dtype: str,
    model_dtype: str,
    axis: int = -1,
    out_shape: Optional[List[tir.PrimExpr]] = None,
    ft_reorder: Optional[bool] = False,
) -> te.Tensor:
    """Convert a quantized uint weight to an unquantized float weight."""
    tir_bin_mask = tir.const((1 << bits) - 1, storage_dtype)
    if out_shape is None:
        out_shape = weight.shape
        out_shape[axis] *= num_elem_per_storage
    axis = axis if axis >= 0 else len(out_shape) + axis
    return te.compute(
        shape=out_shape,
        fcompute=lambda *idx: tir.bitwise_and(
            tir.shift_right(
                weight(*idx[:axis], idx[axis] // num_elem_per_storage, *idx[axis + 1 :]),
                (
                    (
                        (idx[axis] % num_elem_per_storage) % 2 * 4
                        + (idx[axis] % num_elem_per_storage) // 2
                    )
                    * bits
                    if ft_reorder
                    else (idx[axis] % num_elem_per_storage) * bits
                ).astype(storage_dtype),
            ),
            tir_bin_mask,
        ).astype(model_dtype),
    )


def quant_and_pack_fp8x4_e4m3_sm90(
    weight_shape,
    packed_shape,
    scale_shape,
    group_size,
    axis,
    model_dtype,
    storage_dtype,
    quantized_dtype,
):
    vector_length = 4
    vec_quantized_dtype = f"{quantized_dtype}x{vector_length}"
    vec_model_dtype = f"{model_dtype}x{vector_length}"
    num_elem_per_storage = vector_length
    # TODO(csullivan) assert on storage dtype / quantize type bytes == vector length
    assert (
        group_size % vector_length == 0
    ), f"Number of elements in a group must be divisible by fp8 vector length {vector_length}"

    @T.prim_func(private=True)
    def quant_pack(
        A: T.Buffer(weight_shape, model_dtype),
        scale: T.Buffer(scale_shape, model_dtype),
        compute: T.Buffer(
            packed_shape,
            storage_dtype,
        ),
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        for i0, i1 in T.grid(T.int64(weight_shape[0]), T.int64(weight_shape[1] // vector_length)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(
                    A[v_i0, v_i1 : v_i1 + vector_length],
                    scale[v_i0, v_i1 * T.int64(vector_length) // T.int64(group_size)],
                )
                T.writes(compute[v_i0, v_i1 * vector_length])
                compute[v_i0, v_i1] = T.reinterpret(
                    storage_dtype,
                    T.Cast(
                        vec_quantized_dtype,
                        A[v_i0, T.ramp(v_i1 * vector_length, 1, vector_length)]
                        / scale[v_i0, v_i1 * T.int64(vector_length) // T.int64(group_size)],
                    ),
                )

    return quant_pack


def dequant_fp8x4_e4m3_sm90(
    packed_weight_shape,
    scale_shape,
    out_shape,
    group_size,
    axis,
    model_dtype,
    storage_dtype,
    quantized_dtype,
):
    vector_length = 4
    vec_quantized_dtype = f"{quantized_dtype}x{vector_length}"
    vec_model_dtype = f"{model_dtype}x{vector_length}"
    num_elem_per_storage = vector_length

    @T.prim_func(private=True)
    def dequant(
        packed_weight: T.Buffer(packed_weight_shape, storage_dtype),
        scale: T.Buffer(scale_shape, model_dtype),
        dequantize: T.Buffer(out_shape, model_dtype),
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        for i0, i1 in T.grid(T.int64(packed_weight_shape[0]), T.int64(packed_weight_shape[1])):
            with T.block("dequantize"):
                v_i0 = T.axis.spatial(T.int64(packed_weight_shape[0]), i0)
                v_i1 = T.axis.spatial(T.int64(packed_weight_shape[1]), i1)
                T.reads(
                    packed_weight[v_i0, v_i1],
                    scale[v_i0, v_i1 * T.int64(vector_length) // T.int64(group_size)],
                )

                dequantize[v_i0, T.ramp(v_i1 * vector_length, 1, vector_length)] = T.Cast(
                    vec_model_dtype, T.reinterpret(vec_quantized_dtype, packed_weight[v_i0, v_i1])
                ) * T.Broadcast(
                    scale[v_i0, v_i1 * T.int64(vector_length) // T.int64(group_size)], vector_length
                )

    return dequant
