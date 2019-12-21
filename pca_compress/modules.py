import torch
import torch.nn as nn
import torch.nn.functional as F


def wrap_module_projection(module, basis, mean):
    if isinstance(module, nn.Linear):
        return Linear(module, basis, mean)
    raise TypeError('unsupported module type: ' + type(module))


class Linear(nn.Module):
    def __init__(self, wrapped, basis, mean):
        super().__init__()
        # The original linear transformation is Wx + b.
        # We then do dimensionality reduction: ((x*W' + b) - m)*P'.
        # To get back to the original basis, we multiply on the
        # right by the matrix P.
        #
        # The new linear transformation becomes:
        #
        #     x*W'*P'*P + (b - m)*P'*P
        #
        with torch.no_grad():
            w = torch.matmul(basis, wrapped.weight)
            b = wrapped.bias - mean
            b = torch.matmul(basis.transpose(), torch.matmul(basis, b[:, None])).view(-1)
        self.basis = basis
        self.register_buffer('basis', basis)
        self.proj_weight = nn.Parameter(w)
        self.bias = nn.Parameter(b)

    def forward(self, x):
        return F.linear(x, torch.matmul(self.basis.transpose(), self.proj_weight), self.bias)
