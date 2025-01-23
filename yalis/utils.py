import torch.distributed as dist
from torch._dynamo import disable

@disable
def print_rank0(msg):
    if dist.get_rank() == 0:
        print(f"{msg}")
