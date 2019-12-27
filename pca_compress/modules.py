import torch
import torch.nn as nn

from .location import NestedLocation, SequentialLayerLocation


def inject_module(location, model, module):
    next_location = location.next_location(model)
    if not isinstance(module, nn.Sequential) or len(module) != 2 or module[1].bias is None:
        location.set_module(model, module)
        return location
    elif next_location is None:
        location.set_module(model, module)
        return NestedLocation(location, SequentialLayerLocation(0))
    next_module = next_location.get_module(model)
    if _is_1x1_conv(next_module) and _is_1x1_conv(module[1]):
        w1 = module[1].weight.view(module[1].weight.shape[0], -1)
        w2 = next_module.weight.view(next_module.weight.shape[0], -1)
        new_w = torch.matmul(w2, w1).view(w2.shape[0], 1, 1, -1)
        bias = torch.matmul(w2, module[1].bias[:, None]).view(-1)
        if next_module.bias is not None:
            bias += next_module.bias
        new_conv = nn.Conv2d(new_w.shape[1], new_w.shape[0], 1).to(w1.device)
        new_conv.weight.detach().copy_(new_w)
        new_conv.bias.detach().copy_(bias)
        location.set_module(model, module[0])
        next_location.set_module(model, new_conv)
        return location
    elif isinstance(next_module, nn.Linear) and isinstance(module[1], nn.Linear):
        w1 = module[1].weight
        w2 = next_module.weight
        new_w = torch.matmul(w2, w1)
        bias = torch.matmul(w2, module[1].bias[:, None]).view(-1)
        if next_module.bias is not None:
            bias += next_module.bias
        new_linear = nn.Linear(new_w.shape[1], new_w.shape[0]).to(w1.device)
        new_linear.weight.detach().copy_(new_w)
        new_linear.bias.detach().copy_(bias)
        location.set_module(model, module[0])
        next_location.set_module(model, new_linear)
        return location
    else:
        location.set_module(model, module)
        return NestedLocation(location, SequentialLayerLocation(0))


def _is_1x1_conv(module):
    return isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1) and module.stride == 1


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
    result = result.to(module.weight.device)
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
    result = result.to(module.weight.device)
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
    result = result.to(module.weight.device)
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
    result = result.to(module.weight.device)
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
        ).to(module.weight.device)
    elif isinstance(module, nn.Conv2d):
        return nn.Sequential(
            nn.Conv2d(module.in_channels, dim,
                      kernel_size=module.kernel_size,
                      stride=module.stride,
                      padding=module.padding,
                      dilation=module.dilation,
                      groups=module.groups),
            nn.Conv2d(dim, module.out_channels, kernel_size=1),
        ).to(module.weight.device)
    raise TypeError('unsupported module type: ' + str(type(module)))
