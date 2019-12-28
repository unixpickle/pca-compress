import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import inject_module, wrap_module_projection


def project_module(model, location, batches, dim, loss_fn=None, greedy=False, before=False):
    """
    Replace the module at the given location with a
    projected version of the layer.

    If loss_fn is not None, then this relies on both the
    inputs and outputs for the batch. Otherwise, it simply
    uses the inputs.

    Args:
        model: an nn.Module to project from.
        location: a LayerLocation compatible with model.
        batches: an iterator over (input, output) batches.
        dim: the final number of dimensions to project to.
        loss_fn: if specified, the gradient of this loss
          function is used to cause the projection to
          affect the output as little as possible.
        greedy: if True and loss_fn is specified, then
          prune the directions which have the greatest
          positive impact on making the loss higher.
        before: if True, project the layer based on its
          input statistics rather than output statistics.

    Returns:
        A new location for the replaced module, for
          example pointing to the new nn.Conv2d.
        This location may be nested inside of the old one.
    """
    if loss_fn is None:
        if greedy:
            raise ValueError('cannot specify greedy=True with loss_fn=None')
        mean, cov = activation_stats(model, location, batches, before)
    else:
        mean, cov, aj = activation_grad_stats(model, location, batches, loss_fn, before)
    with torch.no_grad():
        basis, sigmas, _ = torch.svd(cov)
        if greedy:
            # Use a linear approximation of the loss
            # function to figure out how much each
            # projection will increase the loss.
            sigmas = -torch.sum(basis * torch.matmul(aj, basis), dim=0)
        basis = basis.permute(1, 0).contiguous()
        indexed_sigmas = enumerate(sigmas.detach().cpu().numpy())
        sorted_sigmas = sorted(indexed_sigmas, key=lambda x: x[1], reverse=True)
        major_indices = [x[0] for x in sorted_sigmas[:dim]]
        major_basis = basis[major_indices]
    old_module = location.get_module(model)
    new_module = wrap_module_projection(old_module, major_basis, mean, before=before)
    return inject_module(location, model, new_module)


def mse_losses(x, y):
    return torch.sum(torch.pow(x-y, 2).view(x.shape[0], -1), dim=1)


def project_module_optimal(model, location, batches, dim,
                           loss_fn=mse_losses,
                           mean_samples=None, proj_samples=None,
                           rounds=2):
    """
    Project the module to minimize the loss between the
    previous outputs and the new, projected outputs.

    It is recommended that you only use this with
    deterministic models, i.e. in eval() mode.

    Args:
        model: an nn.Module to project from.
        location: a LayerLocation compatible with model.
        batches: an iterator over (input, output) batches.
          This should be infinite.
        dim: the final number of dimensions to project to.
        loss_fn: if specified, the gradient of this loss
          function on (old, new) is used to measure the
          error induced by each projection.
        mean_samples: the number of samples to use to
          measure the mean activation. By default, a value
          of dim is used.
        proj_samples: the number of projections to sample
          while computing the approximate Hessian. By
          default, dim ** 2 is used.
        rounds: the number of optimization rounds to try
          to approximate the hessian.

    Returns:
        A new location for the replaced module, for
          example pointing to the new nn.Conv2d.
        This location may be nested inside of the old one.
    """
    def limited_batches(target):
        count = 0
        for inputs, outputs in batches:
            yield (inputs, outputs)
            count += inputs.shape[0]
            if count >= target:
                break
    mean, _ = activation_stats(model, location, limited_batches(mean_samples or dim), False)

    all_projections = []
    all_losses = []
    for inputs, _ in limited_batches(proj_samples or dim ** 2):
        outputs, activations = location.forward(model, inputs)
        projections = torch.randn(inputs.shape[0], activations.shape[1])
        projections /= torch.sqrt(torch.sum(projections * projections, dim=-1, keepdim=True))

        def project_activations(acts):
            expanded_projs = projections
            expanded_mean = mean
            while len(expanded_projs.shape) < len(acts.shape):
                expanded_projs = expanded_projs[..., None]
                expanded_mean = expanded_mean[..., None]
            acts = acts - expanded_mean
            acts = acts - expanded_projs * torch.sum(expanded_projs * acts, dim=1, keepdim=True)
            acts = acts + expanded_mean
            return acts
        new_outputs, _ = location.forward(model, inputs, modify=project_activations)
        losses = loss_fn(outputs, new_outputs)
        all_projections.extend(projections.detach().cpu().numpy())
        all_losses.extend(losses.detach().cpu().numpy())

    all_projections = np.array(all_projections)
    all_losses = np.array(all_losses)
    mat = np.zeros([all_projections.shape[1]] * 2, dtype=all_projections.dtype)
    last_loss = math.inf
    for i in range(rounds):
        # Compute the current gradient of the sum of outer
        # product squared errors with respect to our
        # approximation matrix.
        current_outputs = quadratic_products(mat, all_projections)
        deltas = all_losses - current_outputs
        grad = (all_projections.T @ (all_projections * deltas[:, None]))

        # Compute the optimal step size using the solution
        # to an analytical line search.
        grad_outers = quadratic_products(grad, all_projections)
        sum1 = np.sum(grad_outers * deltas)
        sum2 = np.sum(grad_outers * grad_outers)

        mat += grad * sum1/sum2
        loss = np.mean(deltas * deltas)
        if loss >= last_loss:
            break
        last_loss = loss

        # Uncomment for debugging purposes:
        # print('fitting step', i, np.mean(deltas*deltas))

    # TODO: reuse this code from other projection routine.
    cov = torch.from_numpy(mat).to(mean.device)
    basis, sigmas, _ = torch.svd(cov)
    basis = basis.permute(1, 0).contiguous()
    indexed_sigmas = enumerate(sigmas.detach().cpu().numpy())
    sorted_sigmas = sorted(indexed_sigmas, key=lambda x: x[1], reverse=True)
    major_indices = [x[0] for x in sorted_sigmas[:dim]]
    major_basis = basis[major_indices]
    old_module = location.get_module(model)
    new_module = wrap_module_projection(old_module, major_basis, mean)
    return inject_module(location, model, new_module)


def quadratic_products(matrix, vectors):
    return np.sum(vectors * (vectors @ matrix), axis=-1)


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
