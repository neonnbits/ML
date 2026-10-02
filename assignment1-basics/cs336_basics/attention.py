import torch
import torch.nn as nn
from cs336_basics.softmax import torch_softmax
from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding
def sdpa(Q, K, V, mask=None):
    d_k = K.shape[-1]
    attn_scores = Q @ torch.swapaxes(K, -2, -1) / (d_k ** 0.5)
    if mask is not None:
        attn_scores = torch.where(mask, attn_scores, -torch.inf)
    attns = torch_softmax(attn_scores, -1)
    out = attns @ V
    return out

"""
we first have the embedding matrix X of shape (4,12,64) d_model=64
we multiply X with W_q, W_k, W_v of shape (h.d_k, d_model) which contains
the projection matrices of all heads to get Q, K and V of shape (4,12,64)
then we do a matrix reshape to get (4,12,4,16) then transpose axis -2 and -3
to get (4,4,12,16). we also set up a causal attention mask so that token n 
can only attend to tokens <=n
we can pass in this whole matrix to a sdpa and it can compute the attention
we then get the same shape after which we undo the operation by swapping the 
axes -3 and -2 to get (4,12,4,16) and then reshaping it to get (4,12,64)
we then multiply this matrix with the W_o matrix of shape (64,64) to get
(4,12,64)
"""

class MHA(nn.Module):
    def __init__(self, d_model, num_heads, theta=None, max_seq_len=None, device=None, dtype=None):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        if theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
        else:
            self.rope = None

    def forward(self, in_features, token_positions=None):
        q = self.q_proj(in_features)
        k = self.k_proj(in_features)
        v = self.v_proj(in_features)

        q = torch.unflatten(q, -1, (self.num_heads, self.d_k))
        k = torch.unflatten(k, -1, (self.num_heads, self.d_k))
        v = torch.unflatten(v, -1, (self.num_heads, self.d_k))

        q = torch.transpose(q, -2, -3)
        k = torch.transpose(k, -2, -3)
        v = torch.transpose(v, -2, -3)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(0, in_features.shape[-2], device=in_features.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        seq_len = q.shape[-2]
        mask = ~torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=in_features.device), diagonal=1)
        attn = sdpa(q, k, v, mask)

        attn = torch.transpose(attn, -2, -3)
        attn = torch.flatten(attn, -2, -1)
        out = self.output_proj(attn)

        return out

# if __name__ == "__main__":
#     x = torch.rand(1,12,64)
#     mha = MHA(64,4)
#     out1 = mha(x)

#     y = x.clone()
#     y[:, [0,1]] = y[:, [1,0]]
#     mha2 = MHA(64,4)
#     out2 = mha(y)

#     print("out1",out1)
#     print("out2",out2)

#     print(torch.allclose(out1[:, -1], out2[:, -1]))