import torch
import torch.nn as nn
import torch.nn.functional as F


def wrap_module_projection(module, basis, mean):
    if isinstance(module, nn.Linear):
        return Linear(module, basis, mean)
    elif isinstance(module, nn.Conv2d):
        return Conv2d(module, basis, mean)
    raise TypeError('unsupported module type: ' + str(type(module)))


class Linear(nn.Module):
    def __init__(self, wrapped, basis, mean):
        super().__init__()
        # The original linear transformation is Wx + b.
        #
        # We then do dimensionality reduction:
        #     ((x*W' + b) - m)*P'.
        #
        # To get back to the original basis, we multiply on the
        # right by the matrix P.
        #
        # The new linear transformation becomes:
        #
        #     x*W'*P'*P + (b - m)*P'*P + m
        #
        with torch.no_grad():
            w = torch.matmul(basis, wrapped.weight)
            b = wrapped.bias - mean
            b = torch.matmul(basis, b[:, None]).view(-1)
        self.register_buffer('basis', basis)
        self.register_buffer('mean', mean)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(b)

    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)
        result = torch.matmul(result, self.basis)
        return result + self.mean


class Conv2d(nn.Module):
    def __init__(self, wrapped, basis, mean):
        super().__init__()
        with torch.no_grad():
            w = torch.matmul(basis, wrapped.weight.view(wrapped.weight.shape[0], -1))
            b = wrapped.bias - mean
            b = torch.matmul(basis, b[:, None]).view(-1)
        self.register_buffer('basis', basis)
        self.register_buffer('mean', mean)
        self.weight = nn.Parameter(w.view(-1, *wrapped.weight.shape[1:]))
        self.bias = nn.Parameter(b)

        self.stride = wrapped.stride
        self.padding = wrapped.padding
        self.dilation = wrapped.dilation
        self.groups = wrapped.groups

    def forward(self, x):
        result = F.conv2d(x, self.weight,
                          bias=self.bias,
                          stride=self.stride,
                          padding=self.padding,
                          dilation=self.dilation,
                          groups=self.groups)
        final_shape = (result.shape[0],) + result.shape[2:] + (self.basis.shape[1],)
        result = result.permute(0, 2, 3, 1).contiguous().view(-1, result.shape[1])
        result = torch.matmul(result, self.basis)
        result = result.view(final_shape).permute(0, 3, 1, 2).contiguous()
        return result + self.mean[None, :, None, None]
