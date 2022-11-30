import numpy as np
import torch
from typing import List, Any, Dict, Tuple
from torch import Tensor
from torch.nn.functional import cosine_similarity


def cosine_distance_score(mu1: Tensor, mu2: Tensor):
    return 1 - cosine_similarity(mu1, mu2)


def KL_divergence(mu1: Tensor, log_var1: Tensor, mu2: Tensor, log_var2: Tensor, reduce_axes: Tuple[int] = (-1,)):
    """ returns KL(D_2 || D_1) assuming Gaussian distributions with diagonal covariance matrices, and taking D_1 as reference
    ----
    mu1, mu2, log_var1, log_var2: Tensors of sizes (..., Z) (e.g. (Z), (B, Z))

    """

    log_det = log_var1 - log_var2
    trace_cov = (-log_det).exp()
    mean_diff = (mu1 - mu2) ** 2 / log_var1.exp()
    return 0.5 * ((trace_cov + mean_diff + log_det).sum(reduce_axes) - mu1.shape[-1])


def twin_vae_change_score(model, x_1, x_2, verbose=False):
    if "VAE" in str(model.__class__):
        mu_1, log_var_1 = model.encode(x_1)  # batch, latent_dim
        mu_2, log_var_2 = model.encode(x_2)  # batch, latent_dim

    else:
        assert False, "To be implemented!"

    if verbose:
        print("x_1", type(x_1), len(x_1), x_1[0].shape)  # x 256 torch.Size([3, 32, 32])
        print("mu_1", type(mu_1), len(mu_1), mu_1[0].shape)  #
        print("log_var_1", type(log_var_1), len(log_var_1), log_var_1[0].shape)  #

    # distance = KL_divergence(mu_1, log_var_1, mu_2, log_var_2)
    distance = cosine_distance_score(mu_1, mu_2)

    if verbose: print("distance", type(distance), len(distance), distance[0].shape)

    # convert to numpy
    distance = distance.detach().cpu().numpy()
    if verbose: print("distance", distance.shape)

    return distance


def encode_batch(model, xs, keep_latent_log_var):
    if "VAE" in str(model.__class__):
        with torch.no_grad():
            mus, vars = model.encode(xs, only_mu=(not keep_latent_log_var))  # batch, latent_dim
    else:
        assert False, "To be implemented!"
    return mus, vars


def encode_tile(model, x, keep_latent_log_var):
    var = None
    x = x.unsqueeze(0) # one tile -> into a list
    mus, vars = encode_batch(model, x, keep_latent_log_var)
    mu = mus[0] # back into just one
    if keep_latent_log_var:
        var = vars[0] # back into just one
    return mu, var

def twin_vae_change_score_from_latents(mu_1, mu_2):
    distance = cosine_distance_score(mu_1, mu_2)
    distance = distance.detach().cpu().numpy()
    return distance
