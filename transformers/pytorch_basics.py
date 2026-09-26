import torch
import torch.nn as nn

def torch_softmax(M):
    shifted = M - torch.amax(M, dim=-1, keepdim=True)
    exps = torch.exp(shifted)
    probs = exps / torch.sum(exps, dim=-1, keepdim=True)
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums))
    return probs


class Shift(nn.Module):
    def __init__(self, d, dtype=None, device=None):
        super().__init__()
        std = 2
        data = torch.empty(d, dtype=dtype, device=device)
        self.b = nn.Parameter(torch.nn.init.trunc_normal_(data, 0, std, -3*std, 3*std))

    def forward(self, x):
        return x + self.b

class ShiftNoP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.b = torch.zeros(d)

    def forward(self, x):
        return x + self.b

if __name__ == "__main__":
    # print(torch_softmax(torch.tensor([[1, 2, 3]])))
    # print(torch_softmax(torch.tensor([[1, 2, 3], [1000, 999, 998]])))
    # print(torch_softmax(torch.tensor([[101, 102, 103]])))
    # print(torch_softmax(torch.tensor([[301, 302, 303]])))
    # print(torch_softmax(torch.tensor([[1000, 999, 998]])))
    # print(torch_softmax(torch.tensor([[-1000, -999, -998]])))
    # print(torch_softmax(torch.tensor([[1000, -5]])))
    # print(torch_softmax(torch.rand(2, 3, 4)).sum(dim=-1))
    
    # layer = Shift(4)
    # print(layer(torch.ones(2,3,4)))
    # print(list(layer.parameters()))
    # print(list(layer.state_dict()))
    # print(list(layer.named_parameters()))

    # new_b = {'bias': torch.rand(4)}
    # layer.load_state_dict(new_b)
    # print(list(layer.parameters()))
    # print(list(layer.state_dict()))

    layer = Shift(100_000, torch.float16, "mps")
    print(torch.min(layer.b).item())
    print(torch.max(layer.b).item())
    print(torch.std(layer.b).item())
    print(layer.b.dtype)
    print(layer.b.device)
    out = layer(torch.ones(100_000, dtype=torch.float32, device="mps"))
    print(out.dtype)