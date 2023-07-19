import os
import numpy as np
import torch
import torch.distributed as dist
import torch.utils.dlpack
import tvm
from tvm.script import tir as T
from tvm._ffi import register_func


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


def generate_weight_stationary_distributed_matmul(M, N, K, rank, world_size, split_mode="spatial", skip_comm=False):
    needs_pre_op_ = 0
    needs_post_op_ = 0
    if split_mode == "spatial":
        N_per_rank = N//world_size
        K_per_rank = K
        needs_post_op_ = 1 if not skip_comm else 0
        post_op = "all_gather"
    elif split_mode == "reduce":
        N_per_rank = N
        K_per_rank = K//world_size
        needs_pre_op_ = 1 if not skip_comm else 0
        pre_op = "scatter"
        needs_post_op_ = 1
        post_op = "all_reduce"
    else:
        raise ValueError("Unsupported split mode used")
    
    # RowParallelLinear with skip_comm == False
    if needs_post_op_ and needs_pre_op_:
        @T.prim_func
        def tir_matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            A = T.match_buffer(a, (M, K), dtype="float32")
            B = T.match_buffer(b, (N_per_rank, K_per_rank), dtype="float32")
            C = T.match_buffer(c, (M, N), dtype="float32")
            A_per_rank = T.alloc_buffer((M, K_per_rank), dtype="float32")

            with T.block():
                T.reads(A[0:M, 0:K])
                T.writes(A_per_rank[0:M, 0:K_per_rank])
                T.tvm_call_packed("tvm.torch.distributed.collective", pre_op, rank, world_size, A, A_per_rank)

                for i0, j0, k0 in T.grid(M, N_per_rank, K_per_rank):
                    with T.block("matmul"):
                        i, j, k = T.axis.remap("SSR", [i0, j0, k0])
                        with T.init():
                            C[i, j] = 0.0
                        C[i, j] += A_per_rank[i, k] * B[j, k]
                with T.block("comm"):
                    T.reads(C[0:M, 0:N_per_rank])
                    T.writes(C[0:M, 0:N])
                    T.tvm_call_packed("tvm.torch.distributed.collective", post_op, rank, world_size, C)
                
    # ColumnParallelLinear with skip_comm == False or RowParallelLinear with skip_comm == True
    elif needs_post_op_:
        @T.prim_func
        def tir_matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            A = T.match_buffer(a, (M, K_per_rank), dtype="float32")
            B = T.match_buffer(b, (N_per_rank, K_per_rank), dtype="float32")
            C = T.match_buffer(c, (M, N), dtype="float32")

            with T.block():
                for i0, j0, k0 in T.grid(M, N_per_rank, K_per_rank):
                    with T.block("matmul"):
                        i, j, k = T.axis.remap("SSR", [i0, j0, k0])
                        with T.init():
                            C[i, j] = 0.0
                        C[i, j] += A[i, k] * B[j, k]
                        
                with T.block("comm"):
                    T.reads(C[0:M, 0:N_per_rank])
                    T.writes(C[0:M, 0:N])
                    T.tvm_call_packed("tvm.torch.distributed.collective", post_op, rank, world_size, C_per_rank, C)
    # ColumnParallelLinear with skip_comm == True
    else:
        @T.prim_func
        def tir_matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            A = T.match_buffer(a, (M, K), dtype="float32")
            B = T.match_buffer(b, (N_per_rank, K_per_rank), dtype="float32")
            C = T.match_buffer(c, (M, N_per_rank), dtype="float32")
            
            
            for i0, j0, k0 in T.grid(M, N_per_rank, K_per_rank):
                with T.block("matmul"):
                    i, j, k = T.axis.remap("SSR", [i0, j0, k0])
                    with T.init():
                        C[i, j] = 0.0
                    C[i, j] += A[i, k] * B[j, k]                      
    
                  
    return tir_matmul

@tvm.register_func("tvm.torch.distributed.collective")
def collective_communication(op: str, rank: int, world_size: int, dltensor):
    if op == "all_reduce":
        tensor = torch.from_dlpack(dltensor)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    elif op == "all_gather":
        raise ValueError("Unimplemented")
    elif op == "scatter":
        pass

    return



def test_matmul_all_reduce_tir_dispatch(rank, world_size):
    # Initialize the torch.distributed process group
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(rank)

    # Compile the matrix multiply layer and split weights among
    # the individual process ranks and attached GPU
    target = tvm.target.Target("cuda")
    M, N, K = (128, 128, 128)
    tir_matmul = generate_weight_stationary_distributed_matmul(M, N, K, rank, world_size, "reduce", skip_comm=False)
    tir_matmul.show()
    sch = tvm.tir.Schedule(tir_matmul)
    matmul_block = sch.get_block("matmul")
    i, j, _ = sch.get_loops(matmul_block)
    sch.bind(j, "blockIdx.x")
    sch.bind(i, "threadIdx.x")
    mod = tvm.build(sch.mod, target=target)
    act_shape = (M, K)
    wgt_shape = (N, K//world_size)
    out_shape = (M, N)
    dev = tvm.cuda(rank)
    act_np = np.ones(act_shape, dtype="float32")
    wgt_np = np.ones(wgt_shape, dtype="float32")
    act = tvm.nd.array(act_np, device=dev)
    wgt = tvm.nd.array(wgt_np, device=dev)
    out = tvm.nd.empty(out_shape, device=dev)
    mod(act, wgt, out)
        
def main():
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # test_matmul_all_reduce(rank, world_size)
    # test_matmul_all_gather(rank, world_size)
    test_matmul_all_reduce_tir_dispatch(rank, world_size)


if __name__ == "__main__":
    main()
