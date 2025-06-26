import torch_directml
import torch

dml = torch_directml.device()

x = torch.randn(3, 3).to(dml)
print(x)
