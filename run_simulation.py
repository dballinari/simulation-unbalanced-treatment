from simulation.dgps import sim_outcomes, sim_covariates, propensity_scores
from simulation.estimator import estimate_ate

import argparse
import numpy as np
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt

# parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--num_simulations', type=int, default=100)
argparser.add_argument('--n', type=int, default=1000)
argparser.add_argument('--n_policy', type=int, default=10000)
argparser.add_argument('--p', type=int, default=10)
argparser.add_argument('--alpha', type=float, default=1/4)
argparser.add_argument('--beta', type=int, default=2)
argparser.add_argument('--gamma', type=int, default=4)
argparser.add_argument('--true_ate', type=float, default=1.0)
argparser.add_argument('--cate_type', type=str, default='complex')
argparser.add_argument('--n_estimators', type=int, default=100)
argparser.add_argument('--seed', type=int, default=123)
argparser.add_argument('--n_jobs', type=int, default=None)
argparser.add_argument('--min_samples_leaf', type=int, default=1)
args = argparser.parse_args()


def plot_estimator_distribution(estimator: np.ndarray, true_value: float, ax: plt.Axes, title: str):
    mean_estimator = np.nanmean(estimator)
    std_estimator = np.nanstd(estimator)
    x = np.linspace(-5*std_estimator + mean_estimator, 5*std_estimator + mean_estimator, 100)
    ax.hist(estimator, bins=50, alpha=0.5, density=True)
    ax.plot(x, 1/np.sqrt(2*np.pi*std_estimator)*np.exp(-((x-mean_estimator)/std_estimator)**2/2), 'r--')
    ax.axvline(true_value, color='k', linestyle='dashed', linewidth=1)
    ax.set_title(title)

def evaluate_estimation(ate: np.ndarray, ate_true: float) -> dict:
    bias = ate - ate_true
    return {
        'mean_bias': np.nanmean(bias),
        'rmse': np.nanstd(bias),
        'mae': np.nanmean(np.abs(bias)),
        'std_estimate': np.nanstd(ate),
    }

def evaluate_regret(regrets):
    return {
        'mean_regret': np.nanmean(regrets),
        'std_regret': np.nanstd(regrets),
        'median_regret': np.nanmedian(regrets),
    }


if __name__=='__main__':
    # set seed
    np.random.seed(args.seed)
    estimates_ate = np.zeros(args.num_simulations)
    estimates_ate_under = np.zeros(args.num_simulations)
    estimates_ate_under_all = np.zeros(args.num_simulations)
    regrets = np.zeros(args.num_simulations)
    regrets_under = np.zeros(args.num_simulations)
    regrets_under_all = np.zeros(args.num_simulations)
    proportion_treated = np.zeros(args.num_simulations)
    # define export file names
    file_name = f'{args.num_simulations}_{args.n}_{args.n_policy}_{args.p}_{args.alpha}_{args.beta}_{args.gamma}_{args.true_ate}_{args.cate_type}_{args.n_estimators}_{args.seed}'
    # add progress bar
    progress_bar = tqdm(total=args.num_simulations)
    for i in range(args.num_simulations):
        # simulate data
        x, w, y = sim_outcomes(n=args.n, p=args.p, alpha=args.alpha, beta=args.beta, gamma=args.gamma, true_ate=args.true_ate, cate_type=args.cate_type)
        # in the first simulation, plot the propensity scores
        if i == 0:
            ps = propensity_scores(x, args.alpha, args.beta, args.gamma)
            fig, ax = plt.subplots()
            ax.hist(ps, bins=20, alpha=0.5, density=False)
            ax.grid()
            ax.set_title('Propensity scores')
            fig.savefig(f'results/propensity_scores_{file_name}.png')
        # simulate data for policy
        x_policy = sim_covariates(n=args.n_policy, p=args.p)
        # save proportion of treated
        proportion_treated[i] = np.mean(w)
        # estimate ATE
        ate, regret = estimate_ate(y, w, x, x_policy, args.true_ate, cate_type=args.cate_type, n_estimators=args.n_estimators, random_state=args.seed, n_jobs=args.n_jobs, min_samples_leaf=args.min_samples_leaf)
        # estimate ATE with under sampling only training data
        ate_under, regret_under = estimate_ate(y, w, x, x_policy, args.true_ate, cate_type=args.cate_type, under_sample_train=True, n_estimators=args.n_estimators, random_state=args.seed, n_jobs=args.n_jobs, min_samples_leaf=args.min_samples_leaf)
        # estimate ATE with under sampling both training and test data
        ate_under_all, regret_under_all = estimate_ate(y, w, x, x_policy, args.true_ate, cate_type=args.cate_type, under_sample_train=True, under_sample_test=True, n_estimators=args.n_estimators, random_state=args.seed, n_jobs=args.n_jobs, min_samples_leaf=args.min_samples_leaf)
        # save ate estimates
        estimates_ate[i] = ate
        estimates_ate_under[i] = ate_under
        estimates_ate_under_all[i] = ate_under_all
        # save regrets
        regrets[i] = regret
        regrets_under[i] = regret_under
        regrets_under_all[i] = regret_under_all
        # update progress bar
        progress_bar.update(1)
    # close progress bar
    progress_bar.close()
    # compute summary statistics of biases removing nan
    stats_ate = evaluate_estimation(estimates_ate, args.true_ate)
    stats_ate_under = evaluate_estimation(estimates_ate_under, args.true_ate)
    stats_ate_under_all = evaluate_estimation(estimates_ate_under_all, args.true_ate)
    # compute summary statistics of regrets removing nan
    stats_regret = evaluate_regret(regrets)
    stats_regret_under = evaluate_regret(regrets_under)
    stats_regret_under_all = evaluate_regret(regrets_under_all)
    mean_proportion_treated = np.mean(proportion_treated)
    std_proportion_treated = np.std(proportion_treated)
    # ensure that the results folder exists
    os.makedirs('results', exist_ok=True)
    # dump summary statistics of biases to json
    with open(f'results/ate_policy_{file_name}.json', 'w') as f:
        json.dump({
            'stats_ate': stats_ate, 
            'stats_ate_under': stats_ate_under, 
            'stats_ate_under_all': stats_ate_under_all,
            'stats_regret': stats_regret,
            'stats_regret_under': stats_regret_under,
            'stats_regret_under_all': stats_regret_under_all,
            'mean_proportion_treated': mean_proportion_treated,
            'std_proportion_treated': std_proportion_treated,
            }, f, indent=4)
    # plot of standardized biases in one figure
    fig, ax = plt.subplots(ncols=3)
    plot_estimator_distribution(estimates_ate, args.true_ate, ax[0], 'ATE')
    plot_estimator_distribution(estimates_ate_under, args.true_ate, ax[1], 'ATE under')
    plot_estimator_distribution(estimates_ate_under_all, args.true_ate, ax[2], 'ATE under all')
    # save figure to result folder
    fig.savefig(f'results/ate_{file_name}.png')
