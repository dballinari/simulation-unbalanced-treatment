import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple

from simulation.dgps import outcomes_not_treated, outcomes_treated


def regression_prediction(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, **kwargs) -> np.ndarray:
    # fit random forest regression
    model = RandomForestRegressor(**kwargs)
    model.fit(x_train, y_train)
    # predict outcomes
    y_pred = model.predict(x_test)
    return y_pred

def classification_prediction(x_train: np.ndarray, w_train: np.ndarray, x_test: np.ndarray, **kwargs) -> np.ndarray:
    # fit random forest classification
    model = RandomForestClassifier(**kwargs)
    model.fit(x_train, w_train)
    # predict treatment probabilities
    w_pred = model.predict_proba(x_test)[:,1]
    return w_pred

def estimate_ate(y: np.ndarray, w: np.ndarray, x: np.ndarray, x_policy: np.ndarray, true_ate: float, nfolds: int=2, under_sample_train: bool=False, under_sample_test: bool=False, **kwargs) -> float:
    # if both train and test sets are under-sampled, under-sample the entire dataset
    if under_sample_train and under_sample_test:
        y, w, x = _under_sample_majority_treatment(y, w, x)
    # compute pseudo-outcomes
    tau = _estimate_pseudo_outcomes(y, w, x, nfolds, under_sample_fitting=under_sample_train and not under_sample_test,**kwargs)
    # estimate ATE using doubly robust estimator
    ate = np.mean(tau)
    # compute optimal policy and its regret
    w_opt = _compute_optimal_policy(tau-true_ate, x, x_policy, **kwargs) # use the true ATE as the cost for implementing the policy
    regret = _compute_regret(w_opt,x_policy, true_ate)
    return ate, regret


def _estimate_pseudo_outcomes(y: np.ndarray, w: np.ndarray, x: np.ndarray, nfolds: int=2, under_sample_fitting: bool=False, **kwargs) -> np.ndarray:
    # function to estimate pseudo-outcomes using cross-fitting
    # split sample into folds
    n = x.shape[0]
    idx = np.random.choice(np.arange(n), size=n, replace=False)
    idx = np.array_split(idx, nfolds)
    # estimate ration of treated to non-treated
    ratio_treated = np.sum(w)/np.sum(1-w)
    # initialize pseudo-outcomes
    tau = np.zeros(n)
    # loop over folds
    for i in range(nfolds):
        # split sample into train and test
        idx_test = idx[i]
        idx_train = np.concatenate(idx[:i] + idx[(i+1):])
        x_train = x[idx_train,:]
        y_train = y[idx_train]
        w_train = w[idx_train]
        x_test = x[idx_test,:]
        y_test = y[idx_test]
        w_test = w[idx_test]
        # if train and/or test sample have no treated or no non-treated, set tau to nan
        if (np.sum(w_train==1)==0) or (np.sum(w_train==0)==0) or (np.sum(w_test==1)==0) or (np.sum(w_test==0)==0):
            tau[idx_test] = np.nan
            continue
        # under-sample fitting folds if specified
        if under_sample_fitting:
            y_train, w_train, x_train=_under_sample_majority_treatment(y_train, w_train, x_train)
        # predict outcomes using data on the treated
        y_pred_treated = regression_prediction(x_train[w_train==1,:], y_train[w_train==1], x_test, **kwargs)
        # predict outcomes using data on the non-treated
        y_pred_not_treated = regression_prediction(x_train[w_train==0,:], y_train[w_train==0], x_test, **kwargs)
        # predict treatment probabilities
        w_pred = classification_prediction(x_train, w_train, x_test, **kwargs)
        # correct predicted probabilities for under-sampling (Dal Pozzolo et al., 2015)
        if under_sample_fitting:
            if ratio_treated < 1:
                # correct for under-sampling of the treated
                w_pred = ratio_treated*w_pred/(ratio_treated*w_pred - w_pred + 1)
            else:
                # correct for under-sampling of the non-treated
                w_pred = w_pred/((1-w_pred)/ratio_treated - w_pred)
        # compute pseudo-outcomes on test set
        tau[idx_test] = y_pred_treated-y_pred_not_treated + w_test*(y_test-y_pred_treated)/(w_pred+1e-10) + (1-w_test)*(y_test-y_pred_not_treated)/(1-w_pred+1e-10)
    return tau



def _under_sample_majority_treatment(y: np.ndarray, w: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = x.shape[0]
    # under-sample the majority class
    n_treated = np.sum(w)
    n_not_treated = n - n_treated
    if n_treated > n_not_treated:
        # under-sample treated
        idx = np.where(w == 1)[0]
        idx = np.random.choice(idx, size=n_not_treated, replace=False)
        idx = np.concatenate((idx, np.where(w == 0)[0]))
    else:
        # under-sample not treated
        idx = np.where(w == 0)[0]
        idx = np.random.choice(idx, size=n_treated, replace=False)
        idx = np.concatenate((idx, np.where(w == 1)[0]))
    x = x[idx,:]
    y = y[idx]
    w = w[idx]
    return y, w, x


def _compute_optimal_policy(pseudo_outcome: np.ndarray, x_test: np.ndarray, x_policy: np.ndarray, **kwargs) -> np.ndarray:
    # define classification target:
    # 1 if pseudo-outcome is positive, 0 otherwise
    pseudo_outcome_sign = pseudo_outcome > 0
    # define weights for random forest classification
    weight = np.abs(pseudo_outcome)
    # fit random forest classification
    model = RandomForestClassifier(**kwargs)
    model.fit(x_test, pseudo_outcome_sign, sample_weight=weight)
    # predict optimal policy
    w_opt = model.predict(x_policy)
    return w_opt

def _compute_regret(w_opt: np.ndarray, x_policy: np.ndarray, true_ate: bool) -> float:
    y_treated = outcomes_treated(x_policy, true_ate)
    y_not_treated = outcomes_not_treated(x_policy)
    # determine oracle policy (i.e. policy that maximizes the expected outcome)
    w_oracle = y_treated - y_not_treated > true_ate # treat only if individual CATE is larger than true ATE
    # define outcome based on policy
    y_policy = y_treated*w_opt + y_not_treated*(1-w_opt)
    y_oracle = y_treated*w_oracle + y_not_treated*(1-w_oracle)
    # compute regret as the average outcome of the oracle policy minus the average outcome of the optimal policy
    regret = np.mean(y_oracle) - np.mean(y_policy)
    return regret
