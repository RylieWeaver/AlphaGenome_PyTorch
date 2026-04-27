# General

# Torch
import torch
import torch.distributed as dist

# AlphaGenome



def is_dist():
    return dist.is_available() and dist.is_initialized()

def is_rank0():
    return (not is_dist()) or dist.get_rank() == 0

def dist_print(*args, **kwargs):
    if is_rank0():
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


### Custom Distributed Sum ###
# NOTE: Communicates grads in backprop
def dist_sum(tensor):
    return _DistSum.apply(tensor)

class _DistSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        tensor = tensor.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        g = grad_output.contiguous()
        dist.all_reduce(g, op=dist.ReduceOp.SUM)
        return g
    
