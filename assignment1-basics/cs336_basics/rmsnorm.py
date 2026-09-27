import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt((torch.sum(x*x, dim=-1, keepdim=True) * (1/self.d_model)) + self.eps)
        rmsnorm = (x/rms) * self.weight

        return rmsnorm.to(in_dtype)

if __name__ == "__main__":
    layer = RMSNorm(64)
    x = torch.randn((2,5,64))
    result1 = layer(x)
    result2 = layer(x*10)
    print(torch.allclose(result1, result2))