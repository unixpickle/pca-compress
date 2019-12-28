import torch
import torch.nn as nn
import torch.nn.functional as F


def activation_stats(model, location, batches, before):
    """
    Gather the mean and covariance of activations in a
    model at a specific LayerLocation.

    Returns:
        A tuple (mean, covariance).
    """
    module = location.get_module(model)
    total_count = 0
    output_sum = None
    outer_sum = None
    with torch.no_grad():
        for batch, _ in batches:
            outputs = location.layer_values(model, batch, before=before)
            outputs = _combine_spatio_temporal(module, before, outputs)
            outer = torch.matmul(outputs.permute(1, 0), outputs)
            total_count += outputs.shape[0]
            if output_sum is None:
                output_sum = torch.sum(outputs, dim=0)
                outer_sum = outer
            else:
                output_sum += torch.sum(outputs, dim=0)
                outer_sum += outer
        mean = output_sum / float(total_count)
        cov = outer_sum / float(total_count) - torch.matmul(mean[:, None], mean[None])
    return mean, cov


def activation_grad_stats(model, location, batches, loss_fn, before):
    """
    Compute the mean of activations (m) and the inner
    product matrix ((A-m)'*J + J'*(A-m)), where J is the
    gradient of the loss for every activation A.

    Returns:
        A tuple (mean, prod, aj), where aj is (A-m)'*J.
    """
    module = location.get_module(model)
    total_count = 0
    activ_sum = None
    grad_sum = None
    outer_sum = None
    for batch_in, batch_out in batches:
        outputs, befores, afters = location.forward_both(model, batch_in)
        if before:
            activations = befores
        else:
            activations = afters
        activations = _combine_spatio_temporal(module, before, activations.detach())

        # Adjust loss scale to correct for batch size.
        loss = loss_fn(outputs, batch_out) * batch_in.shape[0]
        grads = _combined_gradients(module, befores, afters, loss, before)

        outer = torch.matmul(activations.permute(1, 0), grads)
        total_count += activations.shape[0]
        if activ_sum is None:
            activ_sum = torch.sum(activations, dim=0)
            grad_sum = torch.sum(grads, dim=0)
            outer_sum = outer
        else:
            activ_sum += torch.sum(activations, dim=0)
            grad_sum += torch.sum(grads, dim=0)
            outer_sum += outer
    mean = activ_sum / float(total_count)
    mean_prod = torch.matmul(activ_sum[:, None], grad_sum[None]) / float(total_count ** 2)
    aj = outer_sum / float(total_count) - mean_prod
    cov = aj + aj.permute(1, 0).contiguous()
    return mean, cov, aj


def _combine_spatio_temporal(module, before, outputs):
    if len(outputs.shape) == 2:
        return outputs
    elif not before:
        outputs = outputs.view(outputs.shape[0], outputs.shape[1], -1)
        outputs = outputs.permute(0, 2, 1).contiguous()
        outputs = outputs.view(-1, outputs.shape[2])
        return outputs

    if not isinstance(module, nn.Conv2d):
        raise ValueError('cannot combine spatio-temporal dimensions before layer: ' +
                         str(type(module)))
    patches = F.unfold(outputs, module.kernel_size,
                       dilation=module.dilation,
                       padding=module.padding,
                       stride=module.stride)
    patches = patches.permute(0, 2, 1).contiguous().view(-1, patches.shape[1])
    return patches


def _combined_gradients(module, befores, afters, loss, before):
    if before and isinstance(module, nn.Conv2d):
        upstream = torch.autograd.grad(loss, afters)[0]
        patches = F.unfold(befores.detach(), module.kernel_size,
                           dilation=module.dilation,
                           padding=module.padding,
                           stride=module.stride).requires_grad_(True)
        packed = patches.permute(0, 2, 1).contiguous().view(-1, patches.shape[1])
        outputs = torch.matmul(packed, module.weight.view(
            module.out_channels, -1).transpose(1, 0))
        outputs = outputs.view(patches.shape[0], patches.shape[2], -1)
        outputs = outputs.permute(0, 2, 1).contiguous().view(upstream.shape)
        grads = torch.autograd.grad(outputs, packed, grad_outputs=upstream)[0]
    else:
        if before:
            activations = befores
        else:
            activations = afters
        grads = torch.autograd.grad(loss, activations)[0]
        grads = _combine_spatio_temporal(module, before, grads.detach())
    return grads
