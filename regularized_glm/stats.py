import numpy as np
from statsmodels.api import families


def _weighted_design_matrix_svd(design_matrix, sqrt_penalty_matrix, weights):
    _, R = np.linalg.qr(np.sqrt(weights[:, np.newaxis]) * design_matrix)
    U, singular_values, Vt = np.linalg.svd(
        np.concatenate((R, sqrt_penalty_matrix)), full_matrices=False)
    n_covariates = design_matrix.shape[1]

    # Keep the linearly independent columns using svd
    svd_error = (singular_values.max() * np.max(design_matrix.shape)
                 * np.finfo(singular_values.dtype).eps)
    U = U[:n_covariates, singular_values > svd_error]

    return U, singular_values, Vt


def get_effective_degrees_of_freedom(U):
    '''Trace of the influence matrix'''
    return np.sum(U * U)


def get_coefficient_covariance(U, singular_values, Vt, scale):
    '''Frequentist Covariance Sandwich Estimator'''
    PKt = Vt @ np.diag(1 / singular_values) @ U.T
    return PKt @ PKt.T * scale


def pearson_chi_square(response, predicted_response, prior_weights, variance,
                       degrees_of_freedom):
    residual = response - predicted_response
    residual_degrees_of_freedom = response.shape[0] - degrees_of_freedom
    chi_square = prior_weights * residual ** 2 / variance(predicted_response)
    return np.sum(chi_square) / residual_degrees_of_freedom


def estimate_scale(family, response, predicted_response, prior_weights,
                   degrees_of_freedom):
    if isinstance(family, (families.Binomial, families.Poisson)):
        scale = 1.0
        is_estimated_scale = False
    else:
        scale = pearson_chi_square(
            response, predicted_response, prior_weights, family.variance,
            degrees_of_freedom)
        is_estimated_scale = True
    return scale, is_estimated_scale
