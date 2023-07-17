import os
import numpy as np
import torch
import torch.distributed as dist
import tvm
from tvm.script import tir as T


@T.prim_func
def tir_matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A = T.match_buffer(a, (128, 128), dtype="float32")
    B = T.match_buffer(b, (128, 128), dtype="float32")
    C = T.match_buffer(c, (128, 128), dtype="float32")

    for i0, j0, k0 in T.grid(128, 128, 128):
        with T.block("matmul"):
            i, j, k = T.axis.remap("SSR", [i0, j0, k0])
            with T.init():
                C[i, j] = 0.0
            C[i, j] += A[i, k] * B[j, k]


def test_matmul():
    target = tvm.target.Target("cuda")
    sch = tvm.tir.Schedule(tir_matmul)
    matmul_block = sch.get_block("matmul")
    i, j, k = sch.get_loops(matmul_block)
    sch.bind(j, "blockIdx.x")
    sch.bind(i, "threadIdx.x")
    mod = tvm.build(sch.mod, target=target)
    M, N, K = (128, 128, 128)
    act_shape = (M, K)
    wgt_shape = (N, K)
    out_shape = (M, N)
    dev = tvm.cuda(0)
    act_np = np.ones(act_shape, dtype="float32")
    wgt_np = np.ones(act_shape, dtype="float32")
    act = tvm.nd.array(act_np, device=dev)
    wgt = tvm.nd.array(wgt_np, device=dev)
    out = tvm.nd.empty(out_shape, device=dev)
    mod(act, wgt, out)
    print(out)

   
def main():
    # read_file_and_all_reduce()
    test_matmul()


if __name__ == "__main__":
    main()
