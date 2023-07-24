import numpy as np
from scipy import stats
from typing import Tuple

# Definition of DGPs as in the paper Okasa (2022) [https://arxiv.org/abs/2201.12692] with constant ATE

# Define constants
MIN_COVARIATES = 5

def sim_outcomes(n: int, p: int, alpha: float, beta: int, gamma: int, true_ate: float, invert_dependence: bool=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if p < MIN_COVARIATES:
        raise ValueError(f"Number of covariates must be at least {MIN_COVARIATES}")
    # simulate nxp covariates from a uniform distribution
    x = sim_covariates(n, p)
    # simulate treatment assignment
    w = _sim_treatment_assignment(x, alpha, beta, gamma)
    # simulate outcomes
    y0 = outcomes_not_treated(x)
    y1 = outcomes_treated(x, true_ate, invert_dependence)
    # observed outcomes
    y = y0*(1-w) + y1*w
    return x, w, y

def propensity_scores(x: np.ndarray, alpha: float, beta: int, gamma: int) -> np.ndarray:
    # define sinus function of the product of the first 4 covariates
    f = np.sin(np.prod(x[:,:4], axis=1)*np.pi)
    # propensity scores as beta distribution at f
    ps = alpha*(1 + stats.beta.cdf(f, beta, gamma))
    return ps

def sim_covariates(n: int, p: int) -> np.ndarray:
    # simulate nxp covariates from a uniform distribution
    x = np.random.uniform(size=(n, p))
    return x

def outcomes_not_treated(x: np.ndarray) -> np.ndarray:
    mu = np.sin(np.prod(x[:,:2], axis=1)*np.pi) + 2*(x[:,3]-0.5)**2 + 0.5*x[:,4] + np.random.normal(size=x.shape[0])
    return mu

def outcomes_treated(x: np.ndarray, true_ate: float, invert_dependence: bool=False) -> np.ndarray:
    # invert dependence compared to propensity scores if specified
    if invert_dependence:
        tau = np.cos((1-x[:,0])*np.pi) + np.sin((1-x[:,1])*np.pi) # in expectation equal to 2/pi
    else:
        tau = np.cos(x[:,0]*np.pi) + np.sin(x[:,1]*np.pi) # in expectation equal to 2/pi
    mu = true_ate*np.pi/2*tau + outcomes_not_treated(x) + np.random.normal(size=x.shape[0])
    return mu

def _sim_treatment_assignment(x: np.ndarray, alpha: float, beta: int, gamma: int) -> np.ndarray:
    # simulate treatment assignment
    ps = propensity_scores(x, alpha, beta, gamma)
    # treatment assignment as bernoulli distribution at ps
    w = np.random.binomial(1, ps)
    return w
    