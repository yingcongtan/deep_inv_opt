# Copyright (C) to Yingcong Tan, Andrew Delong, Daria Terekhov. All Rights Reserved.

import torch
import numpy as np

def tensor(data, **kwargs):
    """Returns a torch.DoubleTensor, unlike torch.tensor which returns FloatTensor by default"""
    return torch.DoubleTensor(data, **kwargs)

def detach(*args):
    r = tuple(arg.detach() if isinstance(arg, torch.Tensor) else arg for arg in args)
    return r if len(r) > 1 else r[0]

def as_tensor(*args, **kwargs):
    r = tuple(torch.DoubleTensor(arg, **kwargs) if arg is not None else None for arg in args)
    return r if len(r) > 1 else r[0]

def as_numpy(*args):
    r = tuple(arg.detach().numpy() if isinstance(arg, torch.Tensor) else arg for arg in args)
    return r if len(r) > 1 else r[0]

def as_str(*args, precision=5, indent=0, flat=False):
    r = tuple(np.array2string(as_numpy(x) if not flat else as_numpy(x).ravel(),
                              precision=4, max_line_width=100,
                              threshold=5, floatmode='fixed',
                              suppress_small=True, prefix=' '*indent) if x is not None else 'None'
              for x in args)
    return r if len(r) > 1 else r[0]
  
def build_tensor(x):
    """Builds a tensor from a list of lists x that preserves gradient flow
    through to whatever tensors appear inside x, which does not happen if
    you just call torch.tensor(). Used by ParametricLP base class.

    Useful for writing code like:
    
       a = tensor([1.5], requires_grad=True)
       b = tensor([3.0], requires_grad=True)
    
       x = build_tensor([[a*b, 6.0],
                         [0.0, a+1]])
    
       y = x.sum()
       y.backward()  # gradient of y flows back to a and b
    
       print(a.grad, b.grad)
    
    Builds a tensor out of a list-of-lists representation, but does so using
    torch.cat and torch.stack so that, if any components of the resuling matrix are
    tensors requiring gradients, the final tensor will be dependent on them and als
    require gradients.
    
    If you just use torch.tensor(...) using a list-of-lists, it will simply copy any
    tensors and the resulting tensor will be detached with no gradient required.

    If x is already a tensor, no change. If x is None, returns None.
    """
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    return torch.stack(tuple(torch.cat(tuple(xij.view(-1) if isinstance(xij, torch.Tensor) else tensor((xij,)) for xij in xi)) for xi in x))


def is_feasible(x, A_ub, b_ub, tolerance=0.0):
    return np.all(A_ub @ x <= b_ub + tolerance)  


def enumerate_polytope_vertices(A_ub, b_ub):
    A_ub, b_ub = as_numpy(A_ub, b_ub)
    n, m = A_ub.shape
    assert m == 2, "Only 2D supported"
    vertices = []
    intersect = []

    # check every unique (i, j) combination
    for i in range(n):
        for j in np.arange(i+1, n):
            try:
                # intersect constraints i and j and collect it if it's feasible
                x = np.linalg.inv(A_ub[(i, j), :]) @ b_ub[(i, j), :]
                if is_feasible(x, A_ub, b_ub, 1e-6):
                    vertices.append(x.T)
                    intersect.append((i, j))
            except np.linalg.LinAlgError:
                pass  # if constraints i and j happen to be aligned, skip this pair

    return np.vstack(vertices), np.array(intersect)