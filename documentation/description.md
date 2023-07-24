---
title: "Double-debiased machine learning with unbalanced treatment assignment"
layout: post
date: 2023-07-09 17:00
image: /assets/images/markdown.jpg
headerImage: false
tag:
- Double-debiased machine learning
- Imbalanced classification
- Undersampling
star: false
category: blog
author: danieleballinari
description: Adjusting for unbalanced treatment assignment in double-debiased machine learning
---

## Introduction
In this post, I will explain how to adjust for unbalanced treatment assignment when estimating causal effects using the double-debiased machine learning approach. I will start by introducing the double-debiased estimator and then I will discuss different methods to adjust for unbalanced treatment assignment. In a small simulation study, I then show that the double-debiased estimator is very sensitive to unbalanced treatment assignment and that the proposed methods can help to reduce the bias of the estimator. I extend the simulation to analyse how well these adjustment methods perform for assigning optimal policies.

## Double-debiased ML
In many fields we are interested in measuring the expected effect that a specific policy $D$ has on an outcome variable $Y$. This quantity is called the average treatment effect (ATE) and defined as $E[Y(1)-Y(0)]$, where $Y(1)$ is the potential outcome if we are subject to the policy (treated), and $Y(0)$ if we are not (non-treated). For example, we might be interested to know the effect that a specific drug has on the recovery of a patient. In this case, the drug is the policy and the recovery is the outcome.

The estimation of this quantity is not trivial. In fact, we generally do not observe both potential outcomes for the same subject. Moreover, in many situations we do not have data from a controlled experiment. In order to estimate the ATE, we need to make four assumptions. The first one is the stable unit treatment value assumption (SUTVA), which states that the potential outcome of a subject does not depend on the treatment assignment of other subjects. The second assumption that we need to make is the unconfoundedness assumption. Under this assumption, the treatment assignment is as good as random, conditional on the covariates (i.e. a set of observable characteristics). Third, we need to assume that the treatment assignment is not deterministic. And finally, we assume that the treatment assignment does not affect the covariates (exogenity assumption).

Under these assumptions, we can estimate the ATE using the following formula:
$$
\theta = E[Y(1)-Y(0)] = E[\mu_1(X)-\mu_0(X) + \frac{D(Y-\mu_1(X))}{e(X)} - \frac{(1-D)(Y-\mu_0(X))}{1-e(X)}]
$$
where $e(X)$ is the probability of being treated (propensity score) and $\mu_{(d)}(X)=E[Y|X, D=d]$. The right-hand side of the equation is called the augmented inverse probability weighting (AIPW) estimator or double-robust estimator. Ideally we would like to estimate the nuisance functions $\mu_1(X)$, $\mu_0(X)$ and $e(X)$ using modern machine learning methods while still have a consistent and asymptotically normally distributed estimator. With two crucial ingredients, this is indeed possible. First, the following conditions need to be satisfied[^fn1]:
- Overlap: $\eta < e(x) < 1-\eta$ for $\eta>0$ and for all $x \in \mathcal{X}$.
- Consistency: the machine learning methods are sup-norm consistent:
$$
\sup_{x \in \mathcal{X}} |\hat{\mu}_d(x) - \mu_d(x)| \xrightarrow{p} 0 \quad \text{and} \quad \sup_{x \in \mathcal{X}} |\hat{e}(x) - e(x)| \xrightarrow{p} 0
$$
- Risk decay: the machine learning methods have a risk decay rate that satisfies
$$
E[\hat{\mu}_d(X) - \mu_d(X)]^2 E[\hat{e}(X) - e(X)]^2 = o(n^{-1})
$$
Second, we estimate the ATE using the so-called cross-fitting approach: we split the data randomly into two folds, $\mathcal{I}_1$ and $\mathcal{I}_2$. We then train the machine learning models on $\mathcal{I}_1$ and use them to compute the following quantity on the other fold $\mathcal{I}_2$:
$$
\hat{\tau}_i = \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) + \frac{D_i(Y_i-\hat{\mu}_1(X_i))}{\hat{e}(X_i)} - \frac{(1-D_i)(Y_i-\hat{\mu}_0(X_i))}{1-\hat{e}(X_i)}.
$$
We then repeat the same procedure flipping the folds. We obtain the final estimate of the ATE by averaging the $\hat{\tau}_i$:
$$
\hat{\theta} = \frac{1}{N} \sum_{i=1}^N \hat{\tau}_i
$$
In practice, we can use more than two folds, e.g. use $K$ folds.

For a good explanation of the proof, I recommend the lecture notes of Stefan Wager's course "STATS 361: Causal Inference"[^1]. For a more technical explanation, you can directly refer to the paper on double-debiased machine learning by Chernozhukov et al. (2018).[^2]

## Unbalanced treatment assignment

In many real-world applications, the treatment assignment is not balanced. This means that only very few subjects are treated. In this case, the machine learning model will have a hard time to correctly estimate the propensity scores $e(X)$. In fact, the more extreme the unbalancedness, the more the model will tend to predict a probability of being treated close to zero for all subjects. This is because the model will try to minimize the loss function, which will be dominated by the non-treated subjects. Since the propensity scores appear in the denominator of the AIPW, we will obtain very large weights for the treated subjects. This will lead to a very high variance of the estimator and therefore to a very unstable estimator.

A common approach to address this problem is to undersample the data. This means that we will randomly select a subset of the non-treated subjects and use this subset to train the machine learning model. This will lead to a more balanced training set. However, this approach has two main drawbacks:
- The variance of the estimator will increase, since we are using less data to train the model.
- The estimated propensity scores are not consistent for the original population, but only for the undersampled population.

This problem is well-known in the machine learning literature. In what follows, I will describe the adjustment proposed by Dal Pozzolo et al. (2015).[^3] While they apply their method to the problem of unbalanced classification, it can be easily adapted to the problem of estimating propensity scores. To formalize the concept of undersampling, we can define a random variable $S_i$ which equals 1 if the observation is part of the undersampled data and 0 otherwise. It then follows that $P(S_i=1|D_i=1)=1$ since we keep all treated observations. For the non-treated once, we instead have $P(S_i = 1 | D_i=0) = \gamma < 1$. Notice that since the undersampling technique is not dependent on the covariates $X$, we have $P(S_i=1|D_i=d, X_i)=P(S_i=1|D_i=d)$. So how does undersampling affect the propensity score? This can be easily derived using Bayes' rule:
$$
e_S(X_i) := P(D_i=1|X_i, S_i=1) =  \frac{P(S_i=1|X_i)}{P(S_i=1|X_i) + P(S_i=0|X_i)} = \frac{P(S_i=1|X_i)}{P(S_i=1|X_i) + (1-\gamma)P(S_i=1|X_i)} = \frac{1}{1-\gamma} 
$$

## References

[^1]: Stefan Wager (2020), STATS 361: Causal Inference, Retrieved from [Wager's webpage](https://web.stanford.edu/~swager/teaching.html).

[^2]: Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W. and Robins, J. (2018), Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21: C1-C68. [doi:10.1111/ectj.12097](https://doi.org/10.1111/ectj.12097).

[^3]: A. D. Pozzolo, O. Caelen, R. A. Johnson and G. Bontempi, "Calibrating Probability with Undersampling for Unbalanced Classification," 2015 IEEE Symposium Series on Computational Intelligence, Cape Town, South Africa, 2015, pp. 159-166, [doi: 10.1109/SSCI.2015.33](https://doi.org/10.1109/SSCI.2015.33).