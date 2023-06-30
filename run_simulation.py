from simulation.dgps import sim_outcomes
from simulation.estimator import estimate_ate
import argparse
import numpy as np

argparser = argparse.ArgumentParser()
argparser.add_argument('--num_simulations', type=int, default=100)
argparser.add_argument('--n', type=int, default=1000)
argparser.add_argument('--p', type=int, default=10)
argparser.add_argument('--alpha', type=float, default=0.5)
argparser.add_argument('--beta', type=int, default=1)
argparser.add_argument('--gamma', type=int, default=1)
argparser.add_argument('--true_ate', type=float, default=1.0)
argparser.add_argument('--n_estimators', type=int, default=100)
argparser.add_argument('--seed', type=int, default=123)
args = argparser.parse_args()



if __name__=='__main__':
    # set seed
    np.random.seed(args.seed)
    ate_array = np.zeros(args.num_simulations)
    for i in range(args.num_simulations):
        # simulate data
        x, w, y = sim_outcomes(n=args.n, p=args.p, alpha=args.alpha, beta=args.beta, gamma=args.gamma, true_ate=args.true_ate)
        # estimate ATE
        ate = estimate_ate(y, w, x, n_estimators=args.n_estimators, random_state=args.seed)
        ate_array[i] = ate
    print('ATE: {:.3f} ({:.3f})'.format(np.mean(ate_array), np.std(ate_array))) 