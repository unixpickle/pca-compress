import torch
import torch.nn as nn


def wrap_module_projection(module, basis, mean, before=False):
    if isinstance(module, nn.Linear):
        return wrap_linear(module, basis, mean, before=before)
    elif isinstance(module, nn.Conv2d):
        return wrap_conv(module, basis, mean, before=before)
    raise TypeError('unsupported module type: ' + str(type(module)))


def wrap_linear(module, basis, mean, before=False):
    if before:
        return wrap_linear_before(module, basis, mean)
    else:
        return wrap_linear_after(module, basis, mean)


def wrap_linear_before(module, basis, mean):
    # f(x) = w*(basis'*(basis*(x - mean)) + mean) + b
    #      = w*(basis'*(basis*x - basis*mean) + mean) + b
    #      = (w*basis')*(basis*x - basis*mean) + w*mean + b
    with torch.no_grad():
        w1 = basis
        b1 = -torch.matmul(basis, mean[:, None]).view(-1)
        w2 = torch.matmul(module.weight, basis.permute(1, 0))
        b2 = module(mean[None]).view(-1)
    result = nn.Sequential(
        nn.Linear(module.in_features, basis.shape[0]),
        nn.Linear(basis.shape[0], module.out_features),
    )
    result[0].weight.detach().copy_(w1)
    result[0].bias.detach().copy_(b1)
    result[1].weight.detach().copy_(w2)
    result[1].bias.detach().copy_(b2)
    return result


def wrap_linear_after(module, basis, mean):
    with torch.no_grad():
        w = torch.matmul(basis, module.weight)
        if module.bias is None:
            b = -mean
        else:
            b = module.bias - mean
        b = torch.matmul(basis, b[:, None]).view(-1)
    result = nn.Sequential(
        nn.Linear(module.in_features, basis.shape[0]),
        nn.Linear(basis.shape[0], module.out_features),
    )
    result[0].weight.detach().copy_(w)
    result[0].bias.detach().copy_(b)
    result[1].weight.detach().copy_(basis.permute(1, 0))
    result[1].bias.detach().copy_(mean)
    return result


def wrap_conv(module, basis, mean, before=False):
    if before:
        return wrap_conv_before(module, basis, mean)
    else:
        return wrap_conv_after(module, basis, mean)


def wrap_conv_before(module, basis, mean):
    if module.groups != 1:
        raise NotImplementedError('grouped convolutions are not supported')
    with torch.no_grad():
        w = module.weight.view(module.weight.shape[0], -1)
        w1 = basis.view(basis.shape[0], -1, *module.kernel_size)
        b1 = -torch.matmul(basis, mean[:, None]).view(-1)
        w2 = torch.matmul(w, basis.permute(1, 0)).view(module.out_channels, -1, 1, 1)
        b2 = torch.matmul(w, mean[:, None]).view(-1)
        if module.bias is not None:
            b2 = b2 + module.bias
    result = nn.Sequential(
        nn.Conv2d(module.in_channels,
                  basis.shape[0],
                  module.kernel_size,
                  stride=module.stride,
                  padding=module.padding,
                  dilation=module.dilation,
                  padding_mode=module.padding_mode),
        nn.Conv2d(basis.shape[0], module.out_channels, 1),
    )
    result[0].weight.detach().copy_(w1)
    result[0].bias.detach().copy_(b1)
    result[1].weight.detach().copy_(w2)
    result[1].bias.detach().copy_(b2)
    return result


def wrap_conv_after(module, basis, mean):
    if module.groups != 1:
        raise NotImplementedError('grouped convolutions are not supported')
    with torch.no_grad():
        w = torch.matmul(basis, module.weight.view(module.weight.shape[0], -1))
        w = w.view(-1, *module.weight.shape[1:])
        if module.bias is None:
            b = -mean
        else:
            b = module.bias - mean
        b = torch.matmul(basis, b[:, None]).view(-1)
    result = nn.Sequential(
        nn.Conv2d(
            module.in_channels,
            basis.shape[0],
            module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            padding_mode=module.padding_mode,
        ),
        nn.Conv2d(
            basis.shape[0],
            module.out_channels,
            1,
        ),
    )
    result[0].weight.detach().copy_(w)
    result[0].bias.detach().copy_(b)
    result[1].weight.detach().copy_(basis.permute(1, 0)[:, :, None, None])
    result[1].bias.detach().copy_(mean)
    return result


def wrap_module_baseline(module, dim):
    if isinstance(module, nn.Linear):
        return nn.Sequential(
            nn.Linear(module.in_features, dim),
            nn.Linear(dim, module.out_features),
        )
    elif isinstance(module, nn.Conv2d):
        return nn.Sequential(
            nn.Conv2d(module.in_channels, dim,
                      kernel_size=module.kernel_size,
                      stride=module.stride,
                      padding=module.padding,
                      dilation=module.dilation,
                      groups=module.groups),
            nn.Conv2d(dim, module.out_channels, kernel_size=1),
        )
    raise TypeError('unsupported module type: ' + str(type(module)))
