import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.randn(100000, device='cuda')
    print('mean on cuda:', x.mean().item())
else:
    print('CUDA not available')