# Double-debiased machine learning with unbalanced treatment assignment

## Double-debiased ML
Let's assume that we are interested in measuring the expected effect that a specific policy $D$ has on an outcome variable $Y$. This quantity is called the average treatment effect (ATE) and defined as $E[Y(1)-Y(0)]$, where $Y(1)$ is the potential outcome if we are subject to the policy (treated), and $Y(0)$ if we are not (non-treated). However, in practice we do not observe the potential outcomes of all subjects and the ATE has to be estimated. Fortunately, under a series of assumptions (SUTVA, unconfoundedness) we can use the following identity to estimate the ATE:
$$
E[Y(1)-Y(0)] = E[\mu(1)-\mu(0) + \frac{D(Y-\mu(1))}{e(X)} - \frac{(1-D)(Y-\mu(1))}{1-e(X)}]
$$
where $e(X)$ is the probability of being treated (propensity score). The right-hand side of the equation is called the augmented inverse probability weighting (AIPW) estimator or double-robust estimator.

