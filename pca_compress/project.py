import torch

from .hessapprox import proj_mse_losses, proj_loss_hessian, proj_loss_hessian_local
from .modules import inject_module, wrap_module_projection
from .stats import activation_stats, activation_grad_stats


def project_module_stats(model, location, batches, dim, loss_fn=None, greedy=False, before=False):
    """
    Use activation and/or gradient statistics to project
    the module at the given location.

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
    return _project_module_basis(model, location, dim, mean, basis, sigmas)


def project_module_hessian(model, location, batches, mean_samples, dim,
                           loss_fn=proj_mse_losses,
                           proj_samples=None,
                           rounds=100,
                           rank_loss=False,
                           local=False,
                           before=False):
    """
    Use an approximate hessian of the loss function for
    activation projections to project the module at the
    given location.

    See proj_loss_hessian() and proj_loss_hessian_local()
    for more information.

    Returns:
        A new location for the replaced module, for
          example pointing to the new nn.Conv2d.
        This location may be nested inside of the old one.
    """
    if local:
        matrix, mean = proj_loss_hessian_local(model, location, batches, mean_samples,
                                               loss_fn=loss_fn,
                                               proj_samples=proj_samples,
                                               rounds=rounds,
                                               before=before)
    else:
        matrix, mean = proj_loss_hessian(model, location, batches, mean_samples,
                                         loss_fn=loss_fn,
                                         proj_samples=proj_samples,
                                         rounds=rounds,
                                         rank_loss=rank_loss,
                                         before=before)
    return project_module_eigen(model, location, dim, mean, matrix, before=before)


def project_module_eigen(model, location, dim, mean, matrix, before=False):
    """
    Use the largest singular vectors of a matrix to
    project the module at the given location.

    Args:
        model: an nn.Module to project from.
        location: a LayerLocation compatible with model.
        dim: the final number of dimensions to project to.
        mean: the mean vector to subtract for projection.
        matrix: the matrix whose spectrum to use.
        before: if True, project the layer based on its
          input statistics rather than output statistics.

    Returns:
        A new location for the replaced module, for
          example pointing to the new nn.Conv2d.
        This location may be nested inside of the old one.
    """
    basis, sigmas, _ = torch.svd(matrix)
    basis = basis.permute(1, 0).contiguous()
    return _project_module_basis(model, location, dim, mean, basis, sigmas)


def _project_module_basis(model, location, dim, mean, basis, sigmas):
    indexed_sigmas = enumerate(sigmas.detach().cpu().numpy())
    sorted_sigmas = sorted(indexed_sigmas, key=lambda x: x[1], reverse=True)
    major_indices = [x[0] for x in sorted_sigmas[:dim]]
    major_basis = basis[major_indices]
    old_module = location.get_module(model)
    new_module = wrap_module_projection(old_module, major_basis, mean)
    return inject_module(location, model, new_module)
