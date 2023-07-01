# Double-debiased machine learning with unbalanced treatment assignment

## Double-debiased ML
Let's assume that we are interested in measuring the expected effect that a specific policy $D$ has on an otucome variable $Y$. This quantity is called the average treatment effect (ATE) and defined as $E[Y(1)-Y(0)]$, where $Y(1)$ is the potential outcome if we are subject to the policy (treated), and $Y(0)$ if we are not (non-treated). However, in practice we do not observe the potential outcomes of all subjects and the ATE has to be estimated.

