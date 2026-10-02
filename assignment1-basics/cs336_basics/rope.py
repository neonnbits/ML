import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        i = torch.arange(0, max_seq_len, device=device)
        k = torch.arange(1, (d_k//2)+1, device=device)

        speeds = 1/(theta ** ((2*k-2)/d_k))
        angle_matrix = torch.outer(i, speeds)

        self.register_buffer('cos_table', torch.cos(angle_matrix), persistent=False)
        self.register_buffer('sin_table', torch.sin(angle_matrix), persistent=False)
    
    def forward(self, x, token_positions):
        pair_1 = x[...,::2]
        pair_2 = x[...,1::2]
        
        cos_values = self.cos_table[token_positions]
        sin_values = self.sin_table[token_positions]
        rotated_1 = pair_1 * cos_values  - pair_2 * sin_values
        rotated_2 = pair_1 * sin_values + pair_2 * cos_values

        M = torch.stack((rotated_1, rotated_2), dim=-1)
        M = torch.flatten(M, -2)

        return M

if __name__ == "__main__":
    layer = RotaryPositionalEmbedding(10000.0, 64, 12)
    layer(torch.rand((2,12,64)), torch.arange(0,12))
