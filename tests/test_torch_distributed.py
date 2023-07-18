import os
import numpy as np
import torch
import torch.distributed as dist
import torch.utils.dlpack
import tvm
from tvm.script import tir as T


def generate_matmul(M, N, K):
    @T.prim_func
    def tir_matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (M, K), dtype="float32")
        B = T.match_buffer(b, (N, K), dtype="float32")
        C = T.match_buffer(c, (M, N), dtype="float32")

        for i0, j0, k0 in T.grid(M, N, K):
            with T.block("matmul"):
                i, j, k = T.axis.remap("SSR", [i0, j0, k0])
                with T.init():
                    C[i, j] = 0.0
                C[i, j] += A[i, k] * B[j, k]
                
    return tir_matmul


def test_matmul_all_reduce(rank, world_size):
    # Initialize the torch.distributed process group
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(rank)

    # Compile the matrix multiply layer and split weights among
    # the individual process ranks and attached GPU
    target = tvm.target.Target("cuda")
    M, N, K = (128, 128, 128//world_size)
    sch = tvm.tir.Schedule(generate_matmul(M, N, K))
    matmul_block = sch.get_block("matmul")
    i, j, _ = sch.get_loops(matmul_block)
    sch.bind(j, "blockIdx.x")
    sch.bind(i, "threadIdx.x")
    mod = tvm.build(sch.mod, target=target)
    act_shape = (M, K)
    wgt_shape = (N, K)
    out_shape = (M, N)
    dev = tvm.cuda(rank)
    act_np = np.ones(act_shape, dtype="float32")
    wgt_np = np.ones(wgt_shape, dtype="float32")
    act = tvm.nd.array(act_np, device=dev)
    wgt = tvm.nd.array(wgt_np, device=dev)
    out = tvm.nd.empty(out_shape, device=dev)
    mod(act, wgt, out)
    
    # All reduce between GPUs with torch.distributed
    tensor = torch.from_dlpack(out)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f'Rank {rank} has tensor:\n {tensor}')

def test_matmul_all_gather(rank, world_size):
    # Initialize the torch.distributed process group
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(rank)

    # Compile the matrix multiply layer and split weights among
    # the individual process ranks and attached GPU
    target = tvm.target.Target("cuda")
    M, N, K = (128, 128//world_size, 128)
    sch = tvm.tir.Schedule(generate_matmul(M, N, K))
    matmul_block = sch.get_block("matmul")
    i, j, _ = sch.get_loops(matmul_block)
    sch.bind(j, "blockIdx.x")
    sch.bind(i, "threadIdx.x")
    mod = tvm.build(sch.mod, target=target)
    act_shape = (M, K)
    wgt_shape = (N, K)
    out_shape = (M, N)
    dev = tvm.cuda(rank)
    act_np = np.ones(act_shape, dtype="float32")
    wgt_np = np.ones(wgt_shape, dtype="float32")
    act = tvm.nd.array(act_np, device=dev)
    wgt = tvm.nd.array(wgt_np, device=dev)
    out = tvm.nd.empty(out_shape, device=dev)
    mod(act, wgt, out)
    
    # All gather and concat between GPUs with torch.distributed
    tensor = torch.from_dlpack(out)
    #dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    #print(f'Rank {rank} has tensor:\n {tensor}')
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    tensor_list[rank] = tensor
    torch.distributed.all_gather(tensor_list, tensor)

    # Note: torch.cat already creates a contiguous tensor.
    last_dim = tensor.dim() - 1
    output = torch.cat(tensor_list, dim=last_dim).contiguous()
    print(f'Rank {rank} has tensor:\n {output}\nShape:{output.shape}\n')

        
def main():
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # test_matmul_all_reduce(rank, world_size)
    test_matmul_all_gather(rank, world_size)


if __name__ == "__main__":
    main()
