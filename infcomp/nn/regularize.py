"""
Code credit to Hongseok Namkoong
Ported to Torch + minimal modifications
"""
import numpy as np
import torch
from torch.autograd import Variable
from infcomp.settings import settings


def regularize_expectation(loss):
    u = loss / loss.sum()
    return torch.dot(loss, u)


def regularize_expectation_exp(loss):
    u = loss.exp() / loss.exp().sum()
    return torch.dot(loss, u)


def dynamic_p(z):
    z_centred = z - torch.min(z)
    return z_centred / z_centred.sum()


def regularize(loss, rho=None, tol=1e-10):
    if rho == 0.:
        # ERM
        return loss.mean()
    elif rho is None:
        return torch.dot(loss, dynamic_p(loss))
    # ML: Note that this implementation does NOT give a correct gradient
    z = loss.data.numpy()
    projection = project_onto_chi_square_ball(z, rho, tol)
    return torch.dot(loss, Variable(settings.Tensor(projection)))


def project_onto_chi_square_ball(z, rho, tol=1e-10):
    assert (rho > 0.)
    # Correct errata
    z = -z
    # Normalise
    z = z - z.mean()
    # Sort in increasing order
    z_sort = np.sort(z)

    z_sort_cumsum = z_sort.cumsum()
    z_sort_sqr_cumsum = np.square(z_sort).cumsum()
    nn = float(z_sort.size)

    # Correct errata
    rho = rho / nn

    # If the vector is all zeros (zero empirical variance), return 1/n \mathbf{1}
    if z_sort[0] == 0:
        return np.full(z_sort.size, 1/nn)

    lam_min = 0.

    lam_init_max = max(np.linalg.norm(z_sort, ord=float("inf")) * nn,
                       np.sqrt(nn / (2 * rho)) * np.linalg.norm(z_sort, ord=2))
    lam_max = lam_init_max

    # bisect on lambda to find the optimal lambda value
    while abs(lam_max - lam_min) > tol * lam_init_max:
        lam = .5 * (lam_max + lam_min)
        eta, ind = find_shift_vector_onto_simplex(z_sort, lam, z_sort_cumsum)

        # compute norm(p(lam))^2 * (1+lam * nn)^2
        deriv = .5 * z_sort_sqr_cumsum[ind - 1] / (lam * lam) + \
                .5 * (eta * eta)*ind + \
                z_sort_cumsum[ind - 1] * eta / lam + \
                .5 * (nn - ind) / (nn * nn) - \
                rho / nn
        if deriv > 0.:
            # constraint infeasible, increase lam (dual var)
            lam_min = lam
        else:
            # constraint loose, decrease lam (dual var)
            lam_max = lam

    lam = .5 * (lam_max + lam_min)
    eta, ind = find_shift_vector_onto_simplex(z_sort, lam, z_sort_cumsum)
    ret = np.full(z.size, 1./nn) - z / lam - eta
    ret[ret < 0.] = 0.
    return ret


def find_shift_vector_onto_simplex(z, lam, z_sum):
    low_ind = 1
    high_ind = z_sum.size
    nn = float(high_ind)

    def eta(i):
        # Given i \in [1, len(z)] returns eta
        return 1./nn - (1. + z_sum[i-1] / lam)/(float(i))

    def s(i):
        return (i * z[i - 1] - z_sum[i-1]) / lam

    # Case when \eta does not exist return the last one.
    if s(high_ind) < 1.:
        return eta(high_ind), high_ind

    # It's always going to exist
    while True:
        i = (low_ind + high_ind) // 2
        if s(i) < 1:
            if s(i + 1) >= 1.:
                return eta(i), i
            else:
                low_ind = i + 1
        else:
            high_ind = i - 1
