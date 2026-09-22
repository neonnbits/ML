import numpy as np

"""
matrix of embedding vectors of a token sequence (n, d) 
where n is the total number of tokens and d is the embedding vector dimension

we have three weight matrices W_q (query matrix with shape (d, d_q)), 
W_k(key matrix with shape (d, d_k)), W_v(value matrix with shape (d, d_v))
we first matmul embedding vector matrix X with W_q to obtain query matrix of shape (n, d_q)
then we matmul embedding vector matrix X with W_k to obtain key matrix of shape (n, d_k)
now the query matrix defines what each token is searching for and the key matrix
defines what each token has to offer. 

then we multiply query matrix Q (n, d_q) with transpose of key matrix (d_k, n) 
and as d_q = d_k we obtain 
attention score matrix of shape (n, n) which is not normalized yet.
each score q_i . k_j is a sum of d_k products. those products have random signs, so they
partly cancel. because of that the typical sum grows like √d_k and not like d_k which we
confirmed with the experiment. softmax exponentiates the scores and big gaps become huge
and might produce output weights that are nearly one-hot and lose useful information from other
tokens. dividing by √d_k brings the typical size back to about 1 for any d_k.
we don't divide it by d_k because it would overshoot; the scores would all be near 0 
and the weights would be nearly uniform, so attention won't pick out anything.


we take softmax of the matrix which is just exponentiating and dividing it by
the sum of exponentiations of the matrix row
to obtain attention weights

we get the value matrix V from X @ W_v with shape (n, d_v)
it contains the actual payload that gets transferred at the end to the output matrix
we multiply the attention weight matrix with the value matrix V to obtain an
output matrix of shape (n,d_v)
"""

def self_attention(X, W_q, W_k, W_v):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    d_k = K.shape[-1]

    attn_scores = (Q @ K.swapaxes(-1,-2)) / np.sqrt(d_k)
    attn_weights = softmax(attn_scores)

    out = attn_weights @ V

    return out


def softmax(M):
    shifted = M - M.max(axis=-1, keepdims=True)
    exps = np.exp(shifted)
    probs = exps / np.sum(exps, axis=-1, keepdims=True)

    assert np.allclose(probs.sum(axis=-1), 1)
    return probs

if __name__ == "__main__":
    print(softmax(np.array([[1, 2, 3]])))
    print(softmax(np.array([[1, 2, 3], [1000, 999, 998]])))
    print(softmax(np.array([[101, 102, 103]])))
    print(softmax(np.array([[301, 302, 303]])))
    print(softmax(np.array([[1000, 999, 998]])))
    print(softmax(np.array([[-1000, -999, -998]])))
    print(softmax(np.array([[1000, -5]])))
    print(softmax(np.random.randn(2, 3, 4)).sum(axis=-1))

    n = 3
    d = 4
    d_k = 2
    d_v = 5

    X = np.random.randn(2, n, d)
    W_q = np.random.randn(d, d_k)
    W_k = np.random.randn(d, d_k)
    W_v = np.random.randn(d, d_v)

    out = self_attention(X, W_q, W_k, W_v)
    assert out.shape == (2, n, d_v)
