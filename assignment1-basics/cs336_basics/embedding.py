import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        weight_data = torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device)
        nn.init.trunc_normal_(
            weight_data,
            mean=0,
            std=1,
            a=-3,
            b=3
        )
        self.weight = nn.Parameter(weight_data)

    def forward(self, token_ids):
        return self.weight[token_ids]

if __name__ == "__main__":
    layer = Embedding(1000, 64)
    print(torch.std(layer.weight).item())
    print(torch.min(layer.weight).item())
    print(torch.max(layer.weight).item())