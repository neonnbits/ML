import torch


def torch_softmax(M, dim):
    shifted = M - torch.amax(M, dim=dim, keepdim=True)
    exps = torch.exp(shifted)
    probs = exps / torch.sum(exps, dim=dim, keepdim=True)
    return probs