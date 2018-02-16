import numpy as np
import scipy.linalg
from statsmodels.api import families

from .stats import (_weighted_design_matrix_svd, get_coefficient_covariance,
                    get_effective_degrees_of_freedom, pearson_chi_square)

_EPS = np.finfo(float).eps


def penalized_IRLS(design_matrix, response, sqrt_penalty_matrix=None,
                   penalty=_EPS, family=families.Gaussian(),
                   max_iterations=25, prior_weights=None,
                   offset=None, tolerance=1E-8):
    if design_matrix.ndim < 2:
        design_matrix = design_matrix[:, np.newaxis]
    n_observations, n_covariates = design_matrix.shape

    if prior_weights is None:
        prior_weights = np.ones((n_observations,), dtype=design_matrix.dtype)

    if offset is None:
        offset = np.zeros((n_observations), dtype=response.dtype)

    if sqrt_penalty_matrix is None:
        sqrt_penalty_matrix = np.eye(n_covariates, dtype=design_matrix.dtype)

    is_converged = False

    predicted_response = family.starting_mu(response)
    linear_predictor = family.link(predicted_response)

    sqrt_penalty_matrix = np.sqrt(penalty) * sqrt_penalty_matrix

    augmented_weights = np.ones((n_covariates,), dtype=response.dtype)
    full_design_matrix = np.concatenate((design_matrix, sqrt_penalty_matrix))
    augmented_response = np.zeros((n_covariates,), dtype=response.dtype)
    coefficients = np.zeros((n_covariates,))

    for _ in range(max_iterations):
        link_derivative = family.link.deriv(predicted_response)
        pseudo_data = (linear_predictor + (response - predicted_response)
                       * link_derivative - offset)
        weights = prior_weights / (family.variance(predicted_response)
                                   * link_derivative ** 2)

        full_response = np.concatenate((pseudo_data, augmented_response))
        full_weights = np.concatenate((np.sqrt(weights), augmented_weights))

        coefficients_old = coefficients.copy()
        coefficients = np.linalg.lstsq(
            full_design_matrix * full_weights[:, np.newaxis],
            full_response * full_weights, rcond=None)[0]

        linear_predictor = offset + design_matrix @ coefficients
        predicted_response = family.link.inverse(linear_predictor)

        # use deviance change instead?
        coefficients_change = np.linalg.norm(coefficients - coefficients_old)
        if coefficients_change < tolerance:
            is_converged = True
            break

    U, singular_values, Vt = _weighted_design_matrix_svd(
        design_matrix, sqrt_penalty_matrix, weights)

    effective_degrees_of_freedom = get_effective_degrees_of_freedom(U)
    if isinstance(family, (families.Binomial, families.Poisson)):
        scale = 1.0
        is_estimated_scale = 0.0
    else:
        scale = pearson_chi_square(
            response, predicted_response, prior_weights, family.variance,
            effective_degrees_of_freedom)
        is_estimated_scale = 1.0
    coefficient_covariance = get_coefficient_covariance(
        U, singular_values, Vt, scale)
    deviance = family.deviance(response, predicted_response, weights, scale)
    aic = (-2 * family.loglike(response, predicted_response, weights)
           + 2 * (effective_degrees_of_freedom + 1)
           + 2 * is_estimated_scale)

    return coefficients, is_converged, coefficient_covariance, aic, deviance


def weighted_least_squares(response, design_matrix, weights):
    '''Fit weighted least squares while handling rank deficiencies

    Uses the rank revealing QR.

    Parameters
    ----------
    response : ndarray, shape (n_observations,)
    design_matrix : ndarray, shape (n_observations, n_covariates)
    weights : ndarray, shape (n_observations,)

    Returns
    -------
    coefficients : ndarray, shape (n_covariates)

    '''
    Q, R, pivots = scipy.linalg.qr(
        design_matrix * weights[:, np.newaxis],
        mode='economic', pivoting=True, check_finite=False)
    z = Q.T @ (response * weights)

    # if rank deficient, keep only the independent columns
    qr_error = (np.abs(R[0, 0]) * np.max(design_matrix.shape)
                * np.finfo(R.dtype).eps)
    is_keep = np.abs(np.diag(R)) > qr_error
    n_covariates = design_matrix.shape[1]

    if np.sum(is_keep) < n_covariates:
        R = R[is_keep, is_keep]
        z = z[is_keep]
        pivots = pivots[is_keep]

    coefficients = np.zeros((n_covariates,))
    coefficients[pivots] = np.linalg.solve(R, z)

    return coefficients, R
