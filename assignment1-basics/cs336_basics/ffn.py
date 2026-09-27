import torch
import torch.nn as nn

from cs336_basics.linear import Linear

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)

    def forward(self, x):
        silu_in = self.w1(x)
        silu_out = silu(silu_in)
        gated = silu_out * self.w3(x)
        swiglu = self.w2(gated)
        return swiglu

def silu(x):
    silu_out = x * torch.sigmoid(x)
    return silu_out

# if __name__ == "__main__":
#     print(silu(torch.tensor([-1,0,10])))
#     layer = FFN(64, 128)
#     out = layer(torch.rand((2,5,64)))
#     print(out.shape)
#     print(layer.state_dict().keys())