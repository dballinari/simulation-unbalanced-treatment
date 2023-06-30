import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

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

def estimate_ate(y: np.ndarray, w: np.ndarray, x: np.ndarray, **kwargs) -> float:
    # split sample into train and test
    n = x.shape[0]
    n_train = int(n*0.5)
    x_train = x[:n_train,:]
    x_test = x[n_train:,:]
    y_train = y[:n_train]
    w_train = w[:n_train]
    y_test = y[n_train:]
    w_test = w[n_train:]
    # predict outcomes using data on the treated
    y_pred_treated = regression_prediction(x_train[w_train==1,:], y_train[w_train==1], x_test, **kwargs)
    # predict outcomes using data on the non-treated
    y_pred_not_treated = regression_prediction(x_train[w_train==0,:], y_train[w_train==0], x_test, **kwargs)
    # predict treatment probabilities
    w_pred = classification_prediction(x_train, w_train, x_test)
    # estimate ATE using doubly robust estimator
    ate = np.mean(y_pred_treated-y_pred_not_treated + w_test*(y_test-y_pred_treated)/(w_pred+1e-10) + (1-w_test)*(y_test-y_pred_not_treated)/(1-w_pred+1e-10))
    return ate