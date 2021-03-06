import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .stats import activation_stats


def proj_mse_losses(x, y):
    """
    Compute the MSE loss across all the dimensions except
    the outer (batch) dimension.
    """
    return torch.mean(torch.pow(x-y, 2).view(x.shape[0], -1), dim=1)


def proj_kl_losses(x, y):
    """
    Compute the bidirectional KL loss as:

        (KL(x||y) + KL(y||x))

    where x and y are logit vectors.
    """
    x = F.log_softmax(x, dim=1)
    y = F.log_softmax(y, dim=1)
    x = x.view(x.shape[0], -1)
    y = y.view(x.shape[0], -1)
    kl1 = torch.sum(torch.exp(x)*(x-y), dim=-1)
    kl2 = torch.sum(torch.exp(y)*(y-x), dim=-1)
    return kl1 + kl2


def proj_loss_hessian(model, location, batches, mean_samples,
                      loss_fn=proj_mse_losses,
                      proj_samples=None,
                      rounds=100,
                      rank_loss=False,
                      before=False):
    """
    Compute a Hessian matrix H which approximates the loss
    incurred by projecting out a unit vector x from the
    activation maps at the given location. Thus, x'*H*x is
    the approximate loss.

    It is recommended that you only use this with
    deterministic models, i.e. in eval() mode.

    Args:
        model: an nn.Module to project from.
        location: a LayerLocation compatible with model.
        batches: an iterator over (input, output) batches.
          This should be infinite.
        mean_samples: the number of samples to use to
          measure the mean activation.
        loss_fn: the loss function to approximate. Takes
          the arguments (old_outputs, new_outputs) and
          returns a one-dimensional batch of losses, one
          per batch element.
        proj_samples: the number of projections to sample
          while computing the approximate Hessian. By
          default, activation_channels ** 2 is used.
        rounds: the maximum number of optimization rounds
          for computing the hessian with gradient descent.
        rank_loss: sort losses and use the index as the
          new loss.
        before: if True, project before the activation.

    Returns:
        A tuple (hessian, mean):
          hessian: the approximated hessian.
          mean: the computed mean activation
    """
    mean, _ = activation_stats(model, location, _limited_batches(batches, mean_samples), before)
    dim = mean.shape[0]

    all_projections = []
    all_losses = []
    for inputs, _ in _limited_batches(batches, proj_samples or dim ** 2):
        with torch.no_grad():
            outputs, _ = location.forward(model, inputs, before=before)
            projections = torch.randn(inputs.shape[0], dim).to(mean.device)
            projections /= torch.sqrt(torch.sum(projections * projections, dim=-1, keepdim=True))
            new_outputs = _projected_forward(location, model, inputs, mean, projections, before)
            losses = loss_fn(outputs, new_outputs)
        all_projections.extend(projections.detach().cpu().numpy())
        all_losses.extend(losses.detach().cpu().numpy())

    all_projections = np.array(all_projections)
    all_losses = np.array(all_losses)

    if rank_loss:
        ranked = enumerate(sorted(all_losses))
        l2r = {k: v for v, k in ranked}
        all_losses = np.array([l2r[x] for x in all_losses], dtype=all_losses.dtype)

    mat = torch.zeros(all_projections.shape[1], all_projections.shape[1]).to(mean.device)
    all_projections = torch.from_numpy(all_projections).to(mat.device)
    all_losses = torch.from_numpy(all_losses).to(mat.device)
    last_loss = math.inf
    for i in range(rounds):
        # Compute the current gradient of the sum of outer
        # product squared errors with respect to our
        # approximation matrix.
        current_outputs = _quadratic_products(mat, all_projections)
        deltas = all_losses - current_outputs
        grad = torch.matmul(all_projections.transpose(1, 0), (all_projections * deltas[:, None]))

        # Compute the optimal step size using the solution
        # to an analytical line search.
        grad_outers = _quadratic_products(grad, all_projections)
        sum1 = torch.sum(grad_outers * deltas)
        sum2 = torch.sum(grad_outers * grad_outers)

        mat += grad * sum1 / sum2
        loss = torch.mean(deltas * deltas).item()
        if loss >= last_loss:
            break
        last_loss = loss

        # Uncomment for debugging purposes:
        # print('fitting step', i, last_loss)

    return mat, mean


def proj_loss_hessian_local(model, location, batches, mean_samples,
                            loss_fn=proj_mse_losses,
                            proj_samples=None,
                            rounds=100,
                            before=False):
    """
    Compute a Hessian matrix H which approximates the loss
    incurred by projecting out a unit vector x from the
    activation maps at the given location. Thus, x'*H*x is
    the approximate loss.

    Unlike proj_loss_hessian, this uses second derivatives
    through the model to get more information for every
    sample. Thus, it can require quadratically fewer model
    evaluations.

    Args:
        model: an nn.Module to project from.
        location: a LayerLocation compatible with model.
        batches: an iterator over (input, output) batches.
          This should be infinite.
        mean_samples: the number of samples to use to
          measure the mean activation.
        loss_fn: the loss function to approximate. Takes
          the arguments (old_outputs, new_outputs) and
          returns a one-dimensional batch of losses, one
          per batch element. It should have a minimum
          when the outputs have not changed.
        proj_samples: the number of projections to sample
          while computing the approximate Hessian. By
          default, activation_channels is used.
        rounds: the maximum number of optimization rounds
          for computing the hessian with gradient descent.
        before: if True, project before the activation.

    Returns:
        A tuple (hessian, mean):
          hessian: the approximated hessian.
          mean: the computed mean activation
    """
    mean, _ = activation_stats(model, location, _limited_batches(batches, mean_samples), before)
    dim = mean.shape[0]

    all_inputs = []
    all_outputs = []
    for inputs, _ in _limited_batches(batches, proj_samples or dim):
        projections = torch.randn(inputs.shape[0], mean.shape[0]).to(mean.device)
        projections /= torch.sqrt(torch.sum(projections * projections, dim=-1, keepdim=True))
        zero_projections = torch.zeros_like(projections).requires_grad_(True)

        outputs = _projected_forward(location, model, inputs, mean, projections, before,
                                     dot_directions=zero_projections)
        losses = loss_fn(outputs.detach(), outputs)

        grads = torch.autograd.grad(torch.sum(losses), zero_projections, create_graph=True)[0]
        grads = torch.autograd.grad(torch.sum(grads * projections), zero_projections)[0]

        all_inputs.extend(projections.detach().cpu().numpy())
        all_outputs.extend(grads.detach().cpu().numpy())

    mat = torch.zeros(len(all_inputs[0]), len(all_inputs[0])).to(mean.device)
    all_inputs = torch.from_numpy(np.array(all_inputs)).to(mat.device)
    all_outputs = torch.from_numpy(np.array(all_outputs)).to(mat.device)
    last_loss = math.inf
    for i in range(rounds):
        residual = torch.matmul(all_inputs, mat) - all_outputs
        grad = torch.matmul(all_outputs.transpose(1, 0), residual)
        grad = grad + grad.transpose(1, 0)

        grad_outer = torch.matmul(all_inputs, grad).view(-1)
        step_size = torch.sum(residual.view(-1) * grad_outer) / torch.sum(grad_outer * grad_outer)

        mat -= grad * step_size
        loss = torch.mean(residual * residual).item()
        if loss >= last_loss:
            break
        last_loss = loss

        # Uncomment for debugging purposes:
        # print('fitting step', i, last_loss)

    return mat, mean


def _quadratic_products(matrix, vectors):
    return torch.sum(vectors * torch.matmul(vectors, matrix), dim=-1)


def _limited_batches(batches, target):
    count = 0
    for inputs, outputs in batches:
        yield (inputs, outputs)
        count += inputs.shape[0]
        if count >= target:
            break


def _projected_forward(location, model, inputs, mean, directions, before, dot_directions=None):
    if dot_directions is None:
        dot_directions = directions
    if before and isinstance(location.get_module(model), nn.Conv2d):
        return _projected_forward_expanded(location,
                                           model,
                                           inputs,
                                           mean,
                                           directions,
                                           dot_directions)
    return _projected_forward_simple(location,
                                     model,
                                     inputs,
                                     mean,
                                     directions,
                                     dot_directions,
                                     before)


def _projected_forward_simple(location, model, inputs, mean, directions, dot_directions, before):
    def project_activations(acts):
        expanded_dots = dot_directions
        expanded_dirs = directions
        expanded_mean = mean
        while len(expanded_dirs.shape) < len(acts.shape):
            expanded_dots = expanded_dots[..., None]
            expanded_dirs = expanded_dirs[..., None]
            expanded_mean = expanded_mean[..., None]
        acts = acts - expanded_mean
        term = expanded_dirs * torch.sum(expanded_dots * acts, dim=1, keepdim=True)
        acts = acts - term
        acts = acts + expanded_mean
        return acts

    outputs, _ = location.forward(model, inputs, before=before, modify=project_activations)
    return outputs


def _projected_forward_expanded(location, model, inputs, mean, directions, dot_directions):
    module = location.get_module(model)
    backup = module.forward

    def new_forward(x):
        with torch.no_grad():
            # Hack to avoid doing some integer arithmetic.
            out_shape = backup(x[:1]).shape[1:]

        patches = F.unfold(x, module.kernel_size,
                           dilation=module.dilation,
                           padding=module.padding,
                           stride=module.stride)

        expanded_dots = dot_directions[..., None]
        expanded_dirs = directions[..., None]
        expanded_mean = mean[..., None]

        patches = patches - expanded_mean
        term = expanded_dirs * torch.sum(expanded_dots * patches, dim=1, keepdim=True)
        patches = patches - term
        patches = patches + expanded_mean

        packed = patches.permute(0, 2, 1).contiguous().view(-1, patches.shape[1])
        outputs = torch.matmul(packed, module.weight.view(
            module.out_channels, -1).transpose(1, 0))
        outputs = outputs.view(patches.shape[0], patches.shape[2], -1)
        outputs = outputs.permute(0, 2, 1).contiguous().view(-1, *out_shape)

        if module.bias is not None:
            outputs = outputs + module.bias[..., None, None]
        return outputs

    try:
        module.forward = new_forward
        return model(inputs)
    finally:
        module.forward = backup
