from collections import namedtuple

import numpy as np
import scipy.linalg
from statsmodels.api import families

from .stats import (_weighted_design_matrix_svd, estimate_aic, estimate_scale,
                    get_coefficient_covariance,
                    get_effective_degrees_of_freedom)

_EPS = np.finfo(float).eps
Results = namedtuple(
    'Results', ['coefficients', 'is_converged', 'coefficient_covariance',
                'AIC', 'deviance', 'degrees_of_freedom', 'scale'])


def penalized_IRLS(design_matrix, response, sqrt_penalty_matrix=None,
                   penalty=_EPS, family=families.Gaussian(),
                   max_iterations=25, prior_weights=None,
                   offset=None, tolerance=1E-8):
    '''Estimate coefficients and associated statistics of models in the
    exponential family.

    Parameters
    ----------
    design_matrix : ndarray, shape (n_observations, n_covariates)
    response : ndarray, shape (n_observations,)
    sqrt_penalty_matrix : ndarray, optional,
                          shape (n_observations, n_observations)
    penalty : ndarray, optional, shape (n_observations,)
    family : statsmodels.api.family instance, optional
    max_iterations : int, optional
    prior_weights : ndarray, optional, shape (n_observations,)
    offset : ndarray, optional, shape (n_observations,)
    tolerance : float, optional

    Returns
    -------
    coefficients : ndarray, shape (n_covariates,)
    is_converged : bool
    coefficient_covariance : ndarray, shape (n_covariates, n_covariates)
    aic : float
    deviance : float
    degrees_of_freedom : float
    scale : float

    '''
    if design_matrix.ndim < 2:
        design_matrix = design_matrix[:, np.newaxis]
    if response.ndim < 2:
        response = response[:, np.newaxis]

    n_observations, n_covariates = design_matrix.shape

    if prior_weights is None:
        prior_weights = np.ones_like(response)

    if offset is None:
        offset = np.zeros_like(response)

    if sqrt_penalty_matrix is None:
        sqrt_penalty_matrix = np.eye(n_covariates, dtype=design_matrix.dtype)

    is_converged = False

    predicted_response = family.starting_mu(response)
    linear_predictor = family.link(predicted_response)

    sqrt_penalty_matrix = np.sqrt(penalty) * sqrt_penalty_matrix

    augmented_weights = np.ones_like(response[:n_covariates])
    full_design_matrix = np.concatenate((design_matrix, sqrt_penalty_matrix))
    augmented_response = np.zeros_like(response[:n_covariates])
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
        try:
            coefficients = np.linalg.lstsq(
                full_design_matrix * full_weights,
                full_response * full_weights, rcond=None)[0]
        except (np.linalg.LinAlgError, ValueError):
            coefficients *= np.nan
            break

        linear_predictor = offset + design_matrix @ coefficients
        predicted_response = family.link.inverse(linear_predictor)

        # use deviance change instead?
        coefficients_change = np.linalg.norm(coefficients - coefficients_old)
        if coefficients_change < tolerance:
            is_converged = True
            break

    U, singular_values, Vt = _weighted_design_matrix_svd(
        design_matrix, sqrt_penalty_matrix, weights)

    degrees_of_freedom = get_effective_degrees_of_freedom(U)
    scale, is_estimated_scale = estimate_scale(
        family, response, predicted_response, prior_weights,
        degrees_of_freedom)
    coefficient_covariance = get_coefficient_covariance(
        U, singular_values, Vt, scale)
    deviance = family.deviance(
        response, predicted_response, prior_weights, scale)
    log_likelihood = family.loglike(
        response, predicted_response, prior_weights, scale)
    aic = estimate_aic(log_likelihood, degrees_of_freedom)

    return Results(
        coefficients=np.squeeze(coefficients),
        is_converged=is_converged,
        coefficient_covariance=coefficient_covariance,
        AIC=aic,
        deviance=deviance,
        degrees_of_freedom=degrees_of_freedom,
        scale=scale
    )


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
        design_matrix * weights,
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
