import math

import numpy as np
import torch
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
    def limited_batches(target):
        count = 0
        for inputs, outputs in batches:
            yield (inputs, outputs)
            count += inputs.shape[0]
            if count >= target:
                break
    mean, _ = activation_stats(model, location, limited_batches(mean_samples), before)
    dim = mean.shape[0]

    all_projections = []
    all_losses = []
    for inputs, _ in limited_batches(proj_samples or dim ** 2):
        with torch.no_grad():
            outputs, activations = location.forward(model, inputs, before=before)
        projections = torch.randn(inputs.shape[0], activations.shape[1]).to(mean.device)
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

        with torch.no_grad():
            new_outputs, _ = location.forward(model, inputs,
                                              before=before,
                                              modify=project_activations)
        losses = loss_fn(outputs, new_outputs)
        all_projections.extend(projections.detach().cpu().numpy())
        all_losses.extend(losses.detach().cpu().numpy())

    all_projections = np.array(all_projections)
    all_losses = np.array(all_losses)

    if rank_loss:
        ranked = sorted(enumerate(all_losses), key=lambda x: x[1])
        l2r = {k: v for v, k in ranked}
        all_losses = np.array([l2r[x] for x in all_losses], dtype=all_losses.dtype)

    mat = np.zeros([all_projections.shape[1]] * 2, dtype=all_projections.dtype)
    last_loss = math.inf
    for i in range(rounds):
        # Compute the current gradient of the sum of outer
        # product squared errors with respect to our
        # approximation matrix.
        current_outputs = _quadratic_products(mat, all_projections)
        deltas = all_losses - current_outputs
        grad = (all_projections.T @ (all_projections * deltas[:, None]))

        # Compute the optimal step size using the solution
        # to an analytical line search.
        grad_outers = _quadratic_products(grad, all_projections)
        sum1 = np.sum(grad_outers * deltas)
        sum2 = np.sum(grad_outers * grad_outers)

        mat += grad * sum1/sum2
        loss = np.mean(deltas * deltas)
        if loss >= last_loss:
            break
        last_loss = loss

        # Uncomment for debugging purposes:
        # print('fitting step', i, np.mean(deltas*deltas))

    return torch.from_numpy(mat).to(mean.device), mean


def _quadratic_products(matrix, vectors):
    return np.sum(vectors * (vectors @ matrix), axis=-1)
