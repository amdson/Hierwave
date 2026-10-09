"""Numpy reference for the CONVPOT energy (notes/convpot_test.md, "Definition").

    E(z) = sum_p [ U(z_p) + B_p(z) + H_p(z) ]
    U    = a . e(z_p)
    B_p  = sum_{d != centre} e(z_p)^T A_d e(z_{p+d})
    H_p  = v . softplus( sum_d W_d e(z_{p+d}) + b )        (absent when m = 0)

Offsets d = (dy, dx) row-major over (-1, 0, 1)^2, index 4 = centre.  Off-grid
neighbours use E[pad] when pad >= 0, else the zero vector.  Written directly
from the definition, slow and obvious; the kernel and the JAX fit are tested
against it.
"""
import numpy as np

OFFSETS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


def softplus(x):
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def convpot_energy(grid, E, a, A, W, b, v, pad):
    grid = np.asarray(grid)
    E = np.asarray(E, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    rows, cols = grid.shape
    k = E.shape[1]
    m = W.shape[2] if W.ndim == 3 else 0
    zero = np.zeros(k)

    def emb(y, x):
        if 0 <= y < rows and 0 <= x < cols:
            return E[int(grid[y, x])]
        return E[pad] if pad >= 0 else zero

    total = 0.0
    for y in range(rows):
        for x in range(cols):
            ep = E[int(grid[y, x])]
            total += float(a @ ep)
            h = np.zeros(m)
            for d, (dy, dx) in enumerate(OFFSETS):
                en = emb(y + dy, x + dx)
                if d != 4:
                    total += float(ep @ A[d] @ en)
                if m > 0:
                    h += en @ W[d]
            if m > 0:
                total += float(v @ softplus(h + b))
    return total
