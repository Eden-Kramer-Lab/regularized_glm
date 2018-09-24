import numpy as np
import scipy.stats
from statsmodels.api import families


def _weighted_design_matrix_svd(design_matrix, sqrt_penalty_matrix, weights):
    '''

    Parameters
    ----------
    design_matrix : ndarray, shape (n_observations, n_covariates)
    sqrt_penalty_matrix : ndarray, shape (n_observations, n_observations)
    weights : ndarray, shape (n_observations,)

    Returns
    -------
    U : ndarray, shape (n_observations, min(n_observations, n_covariates))
    singular_values : ndarray, shape (min(n_observations, n_covariates), )
    Vt : ndarray, shape (min(n_observations, n_covariates), n_covariates)

    '''
    _, R = np.linalg.qr(np.sqrt(weights) * design_matrix)
    try:
        U, singular_values, Vt = np.linalg.svd(
            np.concatenate((R, sqrt_penalty_matrix)), full_matrices=False)
    except (np.linalg.LinAlgError, ValueError):
        m, n = np.concatenate((R, sqrt_penalty_matrix)).shape
        k = np.min((m, n))
        U = np.zeros((m, k))
        singular_values = np.zeros((k,))
        Vt = np.zeros((k, n))
    n_covariates = design_matrix.shape[1]

    # Keep the linearly independent columns using svd
    svd_error = (singular_values.max() * np.max(design_matrix.shape)
                 * np.finfo(singular_values.dtype).eps)
    is_independent = singular_values > svd_error
    U = U[:n_covariates, :]
    U[:, ~is_independent] = np.nan
    singular_values[~is_independent] = np.nan
    Vt[:, ~is_independent] = np.nan

    return U, singular_values, Vt


def get_effective_degrees_of_freedom(U):
    '''Degrees of freedom that account for regularization.

    Parameters
    ----------
    U : ndarray, shape (n_observations, n_independent)

    Returns
    -------
    effective_degrees_of_freedom : float

    '''
    return np.sum(U * U)


def get_coefficient_covariance(U, singular_values, Vt, scale):
    '''Frequentist Covariance Sandwich Estimator.

    Parameters
    ----------
    U : ndarray, shape (n_observations, n_independent)
    singular_values : ndarray, shape (n_independent, n_independent)
    Vt : ndarray, shape ()
    scale : float

    Returns
    -------
    coefficient_covariance : ndarray, shape (n_independent, n_independent)

    '''
    PKt = Vt @ np.diag(1 / singular_values) @ U.T
    return PKt @ PKt.T * scale


def pearson_chi_square(response, predicted_response, prior_weights, variance):
    '''Pearson’s chi-square statistic.

    Parameters
    ----------
    response : ndarray, shape (n_observations,)
    predicted_response : ndarray, shape (n_observations,)
    prior_weights : ndarray, shape (n_observations,)
    variance : function
    degrees_of_freedom : int or float

    Returns
    -------
    chi_square : float

    '''
    residual = response - predicted_response
    chi_square = prior_weights * residual ** 2 / variance(predicted_response)
    return np.sum(chi_square)


def estimate_scale(family, response, predicted_response, prior_weights,
                   degrees_of_freedom):
    '''If the scale has to be estimated, the scale is estimated as Pearson's
    chi square.

    Parameters
    ----------
    family : statsmodels.api.families instance
    response : ndarray, shape (n_observations,)
    predicted_response : ndarray, shape (n_observations,)
    prior_weights : ndarray, shape (n_observations,)
    degrees_of_freedom : int or float

    Returns
    -------
    scale : float
    is_estimated_scale : bool

    '''
    if isinstance(family, (families.Binomial, families.Poisson)):
        scale = 1.0
        is_estimated_scale = False
    else:
        residual_degrees_of_freedom = response.shape[0] - degrees_of_freedom
        scale = pearson_chi_square(
            response, predicted_response, prior_weights, family.variance
        ) / residual_degrees_of_freedom
        is_estimated_scale = True
    return scale, is_estimated_scale


def estimate_aic(log_likelihood, degrees_of_freedom):
    '''Akaike information criterion.

    Parameters
    ----------
    log_likelihood : float
    degrees_of_freedom : float

    Returns
    -------
    Akaike_information_criterion : float

    '''
    return -2 * log_likelihood + 2 * degrees_of_freedom


def estimate_aicc(log_likelihood, degrees_of_freedom, n_observations):
    '''AIC accounting for sample size

    Parameters
    ----------
    log_likelihood : float
    degrees_of_freedom : int or float
    n_observations : int

    Returns
    -------
    aicc : float

    '''
    aic = estimate_aic(log_likelihood, degrees_of_freedom)
    sample_size_penalty = (
        2 * degrees_of_freedom ** 2 + 2 * degrees_of_freedom /
        (n_observations - degrees_of_freedom - 1))
    return aic + sample_size_penalty


def estimate_bic(log_likelihood, degrees_of_freedom, n_observations):
    '''Bayesian Information Criterion

    Parameters
    ----------
    log_likelihood : float
    degrees_of_freedom : int or float
    n_observations : int

    Returns
    -------
    bic : float

    '''
    return -2 * log_likelihood + np.log(n_observations) * degrees_of_freedom


def estimate_unbiased_risk_estimator(deviance, n_observations,
                                     degrees_of_freedom, extra_penalty=1):
    '''Scaled AIC.

    Use with Poisson or Binomail.

    Parameters
    ----------
    deviance : float
    n_observations : int
    degrees_of_freedom : float
    extra_penalty : float

    Returns
    -------
    unbiased_risk_estimator : float

    '''
    penalty = 2 * extra_penalty * degrees_of_freedom / n_observations - 1
    return deviance / n_observations + penalty


def estimate_generalized_cross_validation(deviance, n_observations,
                                          degrees_of_freedom, extra_penalty=1):
    '''

    Parameters
    ----------
    deviance : float
    n_observations : int
    degrees_of_freedom : float
    extra_penalty : float

    Returns
    -------
    generalized_cross_validation : float

    '''
    numerator = n_observations * deviance
    denominator = (n_observations - extra_penalty * degrees_of_freedom) ** 2
    return numerator / denominator


def explained_deviance(full_deviance, deviance_func, response, prior_weights,
                       scale):
    '''R_squared for generalized linear models.

    Parameters
    ----------
    full_deviance : float
    deviance_func : function
    response : ndarray, shape (n_observations,)
    prior_weights : ndarray, shape (n_observations,)
    scale : float

    Returns
    -------
    explained_deviance : float

    '''
    null_predicted_response = response.mean() * np.ones_like(response)
    null_deviance = deviance_func(
        response, null_predicted_response, prior_weights, scale)
    return 1.0 - full_deviance / null_deviance


def likelihood_ratio_test(deviance_full, deviance_restricted,
                          degrees_of_freedom_full,
                          degrees_of_freedom_restricted):
    '''Compare goodness of fit of nested models.

    Parameters
    ----------
    deviance_full : float
        Deviance of the more complicated model
    deviance_restricted : float
        Deviance of the simpler model
    degrees_of_freedom_full : float
        Degrees of freedom of the more complicated model
    degrees_of_freedom_restricted : float
        Degrees of freedom of the simpler model
    n_observations : int

    Returns
    -------
    likelihood_ratio : float
    p_value : float

    '''
    degrees_of_freedom = (degrees_of_freedom_restricted
                          - degrees_of_freedom_full)
    likelihood_ratio = deviance_restricted - deviance_full
    p_value = scipy.stats.chi2.sf(likelihood_ratio, df=degrees_of_freedom)

    return likelihood_ratio, p_value


def parametric_bootstrap(coefficients, coefficient_covariance,
                         n_samples=1000):
    '''

    Parameters
    ----------
    coefficients : ndarray, shape (n_coefficients,)
    coefficient_covariance : ndarray, shape (n_coefficients, n_coefficients)
    n_samples : int, optional

    Returns
    -------
    bootstrapped_coefficients : shape (n_coefficients, n_samples)

    '''
    return np.random.multivariate_normal(
        coefficients, coefficient_covariance, n_samples).T
