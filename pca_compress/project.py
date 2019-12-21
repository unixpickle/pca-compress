import torch

from .modules import wrap_module_projection


def project_module(model, location, batches, dim):
    """
    Replace the module at the given location with a
    projected version of the layer.
    """
    with torch.no_grad():
        mean, cov = activation_stats(model, location, batches)
        basis, sigmas = torch.svd(cov)
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
        for batch in batches:
            outputs = location.layer_values(model, batch)
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
