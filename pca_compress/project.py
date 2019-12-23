import torch

from .modules import wrap_module_projection


def project_module(model, location, batches, dim, loss_fn=None, greedy=False):
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
    """
    if loss_fn is None:
        if greedy:
            raise ValueError('cannot specify greedy=True with loss_fn=None')
        mean, cov = activation_stats(model, location, batches)
    else:
        mean, cov, aj = activation_grad_stats(model, location, batches, loss_fn)
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
    new_module = wrap_module_projection(old_module, major_basis, mean)
    location.set_module(model, new_module)


def activation_stats(model, location, batches):
    """
    Gather the mean and covariance of activations in a
    model at a specific LayerLocation.

    Returns:
        A tuple (mean, covariance).
    """
    total_count = 0
    output_sum = None
    outer_sum = None
    with torch.no_grad():
        for batch, _ in batches:
            outputs = location.layer_values(model, batch)
            outputs = _combine_spatio_temporal(outputs)
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


def activation_grad_stats(model, location, batches, loss_fn):
    """
    Compute the mean of activations (m) and the inner
    product matrix ((A-m)'*J + J'*(A-m)), where J is the
    gradient of the loss for every activation A.

    Returns:
        A tuple (mean, prod, aj), where aj is (A-m)'*J.
    """
    total_count = 0
    activ_sum = None
    grad_sum = None
    outer_sum = None
    for batch_in, batch_out in batches:
        outputs, activations = location.forward(model, batch_in)
        loss = loss_fn(outputs, batch_out)
        grads = torch.autograd.grad(loss, activations)[0]
        activations = _combine_spatio_temporal(activations.detach())
        grads = _combine_spatio_temporal(grads.detach())
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


def _combine_spatio_temporal(outputs):
    if len(outputs.shape) > 2:
        outputs = outputs.view(outputs.shape[0], outputs.shape[1], -1)
        outputs = outputs.permute(0, 2, 1).contiguous()
        outputs = outputs.view(-1, outputs.shape[2])
    return outputs
