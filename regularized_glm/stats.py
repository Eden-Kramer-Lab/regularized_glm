import numpy as np


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
    n_observations = response.shape[0]
    residual_degrees_of_freedom = n_observations - degrees_of_freedom
    return (np.sum(prior_weights * residual ** 2 / variance(predicted_response))
            / residual_degrees_of_freedom)
