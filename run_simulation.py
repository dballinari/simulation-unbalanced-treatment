from simulation.dgps import sim_outcomes
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
argparser.add_argument('--p', type=int, default=10)
argparser.add_argument('--alpha', type=float, default=1/4)
argparser.add_argument('--beta', type=int, default=2)
argparser.add_argument('--gamma', type=int, default=4)
argparser.add_argument('--true_ate', type=float, default=1.0)
argparser.add_argument('--n_estimators', type=int, default=100)
argparser.add_argument('--seed', type=int, default=123)
argparser.add_argument('--n_jobs', type=int, default=None)
args = argparser.parse_args()


def plot_bias_distribution(bias: np.ndarray, ax: plt.Axes, title: str):
    mean_bias = np.nanmean(bias)
    std_bias = np.nanstd(bias)
    x = np.linspace(-5, 5, 100)
    ax.hist(bias/std_bias, bins=50, alpha=0.5, density=True)
    ax.plot(x, 1/np.sqrt(2*np.pi)*np.exp(-x**2/2), 'r--', label='N(0,1)')
    ax.axvline(mean_bias/std_bias, color='k', linestyle='dashed', linewidth=1)
    ax.set_title(title)

def evaluate_estimation(ate: np.ndarray, ate_true: float) -> dict:
    bias = ate - ate_true
    return {
        'mean_bias': np.nanmean(bias),
        'rmse': np.nanstd(bias),
        'mae': np.nanmean(np.abs(bias)),
        'std_estimate': np.nanstd(ate),
    }


if __name__=='__main__':
    # set seed
    np.random.seed(args.seed)
    estimates_ate = np.zeros(args.num_simulations)
    estimates_ate_under = np.zeros(args.num_simulations)
    estimates_ate_under_all = np.zeros(args.num_simulations)
    proportion_treated = np.zeros(args.num_simulations)
    # add progress bar
    progress_bar = tqdm(total=args.num_simulations)
    for i in range(args.num_simulations):
        # simulate data
        x, w, y = sim_outcomes(n=args.n, p=args.p, alpha=args.alpha, beta=args.beta, gamma=args.gamma, true_ate=args.true_ate)
        # save proportion of treated
        proportion_treated[i] = np.mean(w)
        # estimate ATE
        ate, policy = estimate_ate(y, w, x, n_estimators=args.n_estimators, random_state=args.seed, n_jobs=args.n_jobs)
        # estimate ATE with under sampling only training data
        ate_under, policy_under = estimate_ate(y, w, x, under_sample_train=True, n_estimators=args.n_estimators, random_state=args.seed, n_jobs=args.n_jobs)
        # estimate ATE with under sampling both training and test data
        ate_under_all, policy_under_all = estimate_ate(y, w, x, under_sample_train=True, under_sample_test=True, n_estimators=args.n_estimators, random_state=args.seed, n_jobs=args.n_jobs)
        # compute biases
        estimates_ate[i] = ate
        estimates_ate_under[i] = ate_under
        estimates_ate_under_all[i] = ate_under_all
        # update progress bar
        progress_bar.update(1)
    # close progress bar
    progress_bar.close()
    # compute summary statistics of biases removing nan
    stats_ate = evaluate_estimation(estimates_ate, args.true_ate)
    stats_ate_under = evaluate_estimation(estimates_ate_under, args.true_ate)
    stats_ate_under_all = evaluate_estimation(estimates_ate_under_all, args.true_ate)
    mean_proportion_treated = np.mean(proportion_treated)
    std_proportion_treated = np.std(proportion_treated)
    # ensure that the results folder exists
    os.makedirs('results', exist_ok=True)
    # dump summary statistics of biases to json
    with open(f'results/bias_ate_{args.num_simulations}_{args.n}_{args.p}_{args.alpha}_{args.beta}_{args.gamma}_{args.true_ate}_{args.n_estimators}_{args.seed}.json', 'w') as f:
        json.dump({
            'stats_ate': stats_ate, 
            'stats_ate_under': stats_ate_under, 
            'stats_ate_under_all': stats_ate_under_all,
            'mean_proportion_treated': mean_proportion_treated,
            'std_proportion_treated': std_proportion_treated,
            }, f, indent=4)
    # plot of standardized biases in one figure
    x = np.linspace(-5, 5, 100)
    fig, ax = plt.subplots(ncols=3)
    plot_bias_distribution(estimates_ate-args.true_ate, ax[0], 'ATE')
    plot_bias_distribution(estimates_ate_under-args.true_ate, ax[1], 'ATE under')
    plot_bias_distribution(estimates_ate_under_all-args.true_ate, ax[2], 'ATE under all')
    # save figure to result folder
    fig.savefig(f'results/bias_ate_{args.num_simulations}_{args.n}_{args.p}_{args.alpha}_{args.beta}_{args.gamma}_{args.true_ate}_{args.n_estimators}_{args.seed}.png')
