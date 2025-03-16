import torch

fn = torch.compile(torch.square)
x=torch.randn(1000, 1000, device='cuda')
res = fn(x)
