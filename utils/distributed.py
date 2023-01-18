import torch.distributed as dist

def get_rank():
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0