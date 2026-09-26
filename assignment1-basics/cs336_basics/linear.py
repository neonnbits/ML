import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        std = (2/(in_features+out_features)) ** (1/2)
        data = torch.empty((out_features, in_features), dtype=dtype, device=device)
        M = nn.init.trunc_normal_(data, mean=0, std=std, a=-3*std, b=3*std)
        self.weight = nn.Parameter(M)

    def forward(self, x):
        out = torch.matmul(x, self.weight.swapaxes(-1,-2))
        return out

if __name__ == "__main__":
    layer = Linear(10, 20, "mps", torch.float32)
    print(layer(torch.tensor([3,4,1,2,4,6,5,4,8,9], dtype=torch.float32, device="mps")))
