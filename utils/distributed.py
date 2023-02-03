import torch.distributed as dist

def get_rank():
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def gather(tensor, tensor_list=None, group=None):
    if group is None:
        group = dist.group.WORLD
    if is_main_process():
        assert (tensor_list is not None)
        dist.gather(tensor, gather_list=tensor_list, group=group)
    else:
        dist.gather(tensor, dst=0, group=group)