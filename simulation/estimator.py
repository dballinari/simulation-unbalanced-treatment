import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple

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

def estimate_ate(y: np.ndarray, w: np.ndarray, x: np.ndarray, under_sample_train: bool=False, under_sample_test: bool=False, **kwargs) -> float:
    # split sample into train and test
    n = x.shape[0]
    n_train = int(n*0.5)
    x_train = x[:n_train,:]
    x_test = x[n_train:,:]
    y_train = y[:n_train]
    w_train = w[:n_train]
    y_test = y[n_train:]
    w_test = w[n_train:]
    # store number of treated and non-treated in train sample
    n_treated = np.sum(w_train)
    n_not_treated = n_train - n_treated
    ratio_treated = n_treated/n_not_treated

    # if train and/or test sample have no treated or no non-treated, return nan
    if (np.sum(w_train==1)==0) or (np.sum(w_train==0)==0) or (np.sum(w_test==1)==0) or (np.sum(w_test==0)==0):
        return np.nan

    # under-sample train and test sets
    if under_sample_train:
        y_train, w_train, x_train=_under_sample_majority_treatment(y_train, w_train, x_train)
    if under_sample_test:
        y_test, w_test, x_test=_under_sample_majority_treatment(y_test, w_test, x_test)

    # predict outcomes using data on the treated
    y_pred_treated = regression_prediction(x_train[w_train==1,:], y_train[w_train==1], x_test, **kwargs)
    # predict outcomes using data on the non-treated
    y_pred_not_treated = regression_prediction(x_train[w_train==0,:], y_train[w_train==0], x_test, **kwargs)
    # predict treatment probabilities
    w_pred = classification_prediction(x_train, w_train, x_test)
    # correct predicted probabilities for under-sampling (Dal Pozzolo et al., 2015) assuming prior treatment probability are the same in train and test
    if under_sample_train and not under_sample_test:
        if ratio_treated < 1:
            # correct for under-sampling of the treated
            w_pred = ratio_treated*w_pred/(ratio_treated*w_pred - w_pred + 1)
        else:
            # correct for under-sampling of the non-treated
            w_pred = w_pred/((1-w_pred)/ratio_treated - w_pred)
    # estimate ATE using doubly robust estimator
    ate = np.mean(y_pred_treated-y_pred_not_treated + w_test*(y_test-y_pred_treated)/(w_pred+1e-10) + (1-w_test)*(y_test-y_pred_not_treated)/(1-w_pred+1e-10))
    return ate


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