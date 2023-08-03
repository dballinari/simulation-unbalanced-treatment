---
title: "Double-debiased machine learning with unbalanced treatment assignment"
layout: post
date: 2023-08-02 22:48
image: /assets/images/markdown.jpg
headerImage: false
tag:
- causal-ml
- undersampling
- ate
category: blog
author: danieleballinari
description: Adjusting for unbalanced treatment assignment in double-debiased machine learning
width: large
---

## Introduction
In the realm of highly unbalanced classification problems, training machine learning models to predict the minority class can be a daunting task. Interestingly, this issue also extends to causal inference. The popular double-debiased machine learning approach for estimating causal effects is sensitive to unbalanced treatment assignment. In this blog post, I will explore how to address this concern and adjust for unbalanced treatment assignment when employing the double-debiased machine learning approach.

I'll start with a concise review of the double-debiased estimator, highlighting its core principles. Then, I'll delve into different methods for adjusting unbalanced treatment assignment, with a focus on a new approach based on undersampling. Remarkably, this method preserves the favorable asymptotic properties of the double-debiased approach.

To illustrate the impact of unbalanced treatment assignment, I'll present the findings of a small simulation study. This study will underscore the sensitivity of the double-debiased estimator and showcase how the proposed adjustment can mitigate bias effectively.

Furthermore, I'll extend the simulation to evaluate the performance of these adjustment in a decision-making scenarios where we try to find an optimal policy to assign treatments.

The code for this blog post is available on [GitHub](https://github.com/dballinari/simulation-unbalanced-treatment).

## Double-debiased ML
In many fields we are interested in measuring the expected effect that a specific policy $D$ has on an outcome variable $Y$. This quantity is called the average treatment effect (ATE) and defined as $E[Y(1)-Y(0)]$, where $Y(1)$ is the potential outcome if we are subject to the policy (treated), and $Y(0)$ if we are not (non-treated). For example, we might be interested to know the effect that a specific drug has on the recovery of a patient. In this case, the drug is the policy and the recovery is the outcome.

The estimation of this quantity is not trivial. In fact, we generally do not observe both potential outcomes for the same subject. Moreover, in many situations we do not have data from a controlled experiment. In order to estimate the ATE, we need to make four assumptions. The first one is the stable unit treatment value assumption (SUTVA), which states that the potential outcome of a subject does not depend on the treatment assignment of other subjects. The second assumption that we need to make is the unconfoundedness assumption. Under this assumption, the treatment assignment is as good as random, conditional on the covariates (i.e. a set of observable characteristics). Third, we need to assume that the treatment assignment is not deterministic. And finally, we assume that the treatment assignment does not affect the covariates (exogenity assumption).

Under these assumptions, we can estimate the ATE using the following formula:

$$
\theta = E[Y(1)-Y(0)] = E\left[\mu_1(X)-\mu_0(X) + \frac{D(Y-\mu_1(X))}{e(X)} - \frac{(1-D)(Y-\mu_0(X))}{1-e(X)}\right]
$$

where $e(X)$ is the probability of being treated (propensity score) and $\mu_{(d)}(X)=E[Y|X, D=d]$. The right-hand side of the equation is called the augmented inverse probability weighting (AIPW) estimator or double-robust estimator. Ideally we would like to estimate the nuisance functions $\mu_1(X)$, $\mu_0(X)$ and $e(X)$ using modern machine learning methods while still have a consistent and asymptotically normally distributed estimator. With two crucial ingredients, this is indeed possible. First, the following conditions need to be satisfied[^1]:
- Overlap: $\eta < e(x) < 1-\eta$ for $\eta>0$ and for all $x \in \mathcal{X}$.
- Consistency: the machine learning methods are sup-norm consistent

$$
\sup_{x \in \mathcal{X}} |\hat{\mu}_d(x) - \mu_d(x)| \xrightarrow{p} 0 \quad \text{and} \quad \sup_{x \in \mathcal{X}} |\hat{e}(x) - e(x)| \xrightarrow{p} 0
$$

- Risk decay: the machine learning methods have a risk decay rate that satisfies

$$
E\left[\left(\hat{\mu}_d(X) - \mu_d(X)\right)^2\right] E\left[\left(\hat{e}(X) - e(X)\right)^2\right] = o(n^{-1})
$$

Second, we estimate the ATE using a so-called cross-fitting approach: we split the data randomly into two folds, $\mathcal{I}_1$ and $\mathcal{I}_2$. We then train the machine learning models on $\mathcal{I}_1$ and use them to compute the following quantity on the other fold $\mathcal{I}_2$:

$$
\hat{\tau}_i = \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) + \frac{D_i(Y_i-\hat{\mu}_1(X_i))}{\hat{e}(X_i)} - \frac{(1-D_i)(Y_i-\hat{\mu}_0(X_i))}{1-\hat{e}(X_i)}.
$$

We then repeat the same procedure flipping the folds. We obtain the final estimate of the ATE by averaging the $\hat{\tau}_i$:

$$
\hat{\theta} = \frac{1}{N} \sum_{i=1}^N \hat{\tau}_i
$$

In practice, we can use more than two folds, and extend this procedure to $K$ folds.

For a good explanation of the proof of why this estimation approach is in fact consistent and asymptotically normally distributed, I recommend the lecture notes of Stefan Wager's course "STATS 361: Causal Inference"[^1]. For a more technical explanation, you can directly refer to the paper on double-debiased machine learning by Chernozhukov et al. (2018).[^2]

## Unbalanced treatment assignment

In many real-world applications, the treatment assignment is not balanced. This means that only very few subjects are treated. This problem arises, for example, in medical applications where the treatment assignment might be very expensive. In this case, the machine learning model will have a hard time to correctly estimate the propensity scores $e(X)$. In fact, the more extreme the unbalancedness, the more the model will tend to predict a probability of being treated close to zero for all subjects. This is because the model will try to minimize the loss function, which will be dominated by the non-treated subjects. Since the propensity scores appear in the denominator of the doubly-robust estimator, we will obtain very large values for the term:

$$
\frac{D_i(Y_i-\hat{\mu}_1(X_i))}{\hat{e}(X_i)}.
$$

This will lead to a very high variance of the ATE estimator and in finite-samples to a considerable bias.

Luckily, this is a well known issue in machine learning classification problems. A common approach to address this issue is _undersampling_. This means that we will randomly select a subset of the non-treated subjects and use this subset to train the machine learning models and estimate the treatment effect. This will lead to a more balanced dataset. However, undersampling the data results in effectively using less data to estimate $\mu_0(X)$, $e(X)$, and most importantly $\theta$.

Here I will explore an alternative approach which only partially undersamples the dataset. This approach is based on the idea that we can undersample only the data used to estimate the machine learning models, while still using all the data to estimate the treatment effect. This requires however an adjustment of the propensity scores. In what follows, I will describe the adjustment proposed by Dal Pozzolo et al. (2015).[^3] While they apply their method to the problem of unbalanced classification, it can be easily adapted to the problem of estimating propensity scores.

To formalize the concept of undersampling, we can define a random variable $S_i$ which equals 1 if the observation is part of the undersampled data and 0 otherwise. 
It then follows that $P(S_i=1|D_i=1)=1$ since we keep all treated observations. For the non-treated once, we instead have $P(S_i = 1 | D_i=0) = \gamma < 1$. 
Notice that since the undersampling technique is not dependent on the covariates $X$, we have $P(S_i=1|D_i=d, X_i)=P(S_i=1|D_i=d)$. 
So how does undersampling affect the propensity score? This can be easily derived using Bayes' rule:

$$
\begin{aligned}
e_S(X_i) &:= P(D_i=1|X_i, S_i=1)\\[0.5cm] &= \frac{P(S_i=1|D_i=1, X_i)P(D_i=1|X_i)}{P(S_i=1|D_i=1, X_i)P(D_i=1|X_i) + P(S_i=1|D_i=0, X_i)P(D_i=0|X_i)}\\[0.5cm]
&= \frac{P(S_i=1|D_i=1)P(D_i=1|X_i)}{P(S_i=1|D_i=1)P(D_i=1|X_i) + P(S_i=1|D_i=0)P(D_i=0|X_i)}\\[0.5cm]
&= \frac{P(D_i=1|X_i)}{P(D_i=1|X_i) + \gamma P(D_i=0|X_i)}\\[0.5cm]
&= \frac{e(X_i)}{e(X_i) + \gamma (1-e(X_i))}.
\end{aligned}
$$

So when we are using an undersampled dataset to estimate the propensity score, we are in fact not estimating the population propensity score $e(X_i)$, but the propensity score of a balanced dataset $e_S(X_i)$. This is also the reason why we have to estimate the treatment effect on the undersampled dataset: the machine learner will be sup-consistent for the propensity score on the undersampled population.

Fortunately, the above formula does not only show that $e_S(X_i) \neq e(X_i)$, but also directly suggests an adjustment of the estimated propensity score to recover the population propensity score:

$$
e(X_i) = \frac{\gamma e_S(X_i)}{\gamma e_S(X_i) + 1-e_S(X_i)}.
$$

The propensity score $e(X_i)$ can therefore be estimated by plugging in the estimated propensity score $e_S(X_i)$ from the undersampled dataset and an estimate of $\gamma$:

$$
\hat\gamma = \frac{\sum_{i=1}^N D_i}{\sum_{i=1}^N 1-D_i}.
$$

Notice that here we can estimated $\gamma$ using the entire dataset, and not a cross-fitting approach. In fact, as I show below, the ATE estimator obtained by following this approach is still asymptotically normal and consistent. Here a short summary of the estimator:

> **Undersampled-calibrated doubly-robust estimator (UC-DR)**
> 1. Compute $\hat\gamma = \frac{\sum_{i=1}^N D_i}{\sum_{i=1}^N 1-D_i}$
> 2. Split the data randomly into two folds, $\mathcal{I}_1$ and $\mathcal{I}_2$
> 3. Undersample $\mathcal{I}_1$ to obtain a balanced sample $\mathcal{I}_1^S$
> 4. Train the machine learning models on $\mathcal{I}_1^S$ to obtain $\hat{\mu}_0$, $\hat{\mu}_1$ and $\hat{e}_S$.
> 5. Calibrated the propensity score:\
> $$
> \hat{e}(X_i) = \frac{\hat\gamma \hat{e}_S(X_i)}{\hat\gamma \hat{e}_S(X_i) + 1-\hat{e}_S(X_i)}
> $$
> 6. Compute the following quantity on the other fold $\mathcal{I}_2$:\
> $$
\hat{\tau}_i = \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) + \frac{D_i(Y_i-\hat{\mu}_1(X_i))}{\hat{e}(X_i)} - \frac{(1-D_i)(Y_i-\hat{\mu}_0(X_i))}{1-\hat{e}(X_i)}.
> $$
> 7. Repeat the same procedure flipping the folds.
> 8. Obtain the final estimate of the ATE by averaging the $\hat{\tau}_i$:\
> $$
\hat{\theta} = \frac{1}{N} \sum_{i=1}^N \hat{\tau}_i
> $$

You can find a proof of the asymptotic normality of this estimator [at the end](#proof-of-the-asymptotic-properties) of this post.


## A small simulation study
To better understand how the undersampling-calibrated doubly-robust estimator (UC-DR) performs, I run a small simulation study. I use a very similar set-up as in the extensive simulation study of Gabriel Okasa[^4] (2022). If you are not interested in all the details about the data generating process, you can skip to the next section. In brief, I generate data such that the average treatment effect equals 1, only very few observations are actually treated (~2% and ~12%), the potential outcomes and propensity score are highly non-linear, and only four out of the 100 features are actually relevant. The machine learner of choice is a random forest with 200 trees for both classification and regression tasks. The code for the simulation study can be found in my [GitHub repository](https://github.com/dballinari/simulation-unbalanced-treatment).

### Data generating process
The outcome variable is generated as follows:

$$
Y_i = Y_i(1) D_i + Y_i(0) (1-D_i)\\[0.5cm]
$$

with

$$
Y_i(1) = \mu_1(X_i) + \epsilon_i, \quad  \epsilon_i\sim \mathcal{N}(0,1)\\[0.5cm]
Y_i(0) = \mu_0(X_i) + \epsilon_i, \quad  \epsilon_i\sim \mathcal{N}(0,1)\\[0.5cm]
D_i = \mathbb{1}\{e(X_i) > U_i\}, \quad  U_i\sim \mathcal{U}(0,1).
$$

The features $X_i$ are a 100-dimensional vector drawn independently from a uniform distribution on $[0,1]$, that is $X_i \sim \mathcal{U}([0,1]^{100})$. The propensity score is given by:

$$
e(X_i) = k \cdot \left(1 + \beta_{2,4}(f(X_i)) \right)
$$

where $f(x) = \sin(\pi \cdot x_1 \cdot x_2 \cdot x_3 \cdot x_4)$ and $\beta_{2,4}$ is the beta cumulative distribution function with parameter 2 and 4. I set the parameter $k$ such that only either 2% or 12% of the observations are treated. Finally, I define the potential outcomes as follows:

$$
\mu_0(X_i) = \sin(\pi \cdot X_{i,1} \cdot X_{i,2}) + 2 \cdot (X_{i,3} - 0.5)^2 + 0.5 \cdot X_{i,4}\\[0.5cm]
\mu_1(X_i) = \mu_0(X_{i}) + \eta(X_{i,1}) \cdot \eta(X_{i,2})\\[0.5cm]
$$

where the function $\eta(x)$ is defined as:

$$
\eta(x) = 1 + \frac{1}{1 + \exp(-20 \cdot (x - 0.5))} - 0.5.
$$

### Results

Before presenting the results, I will briefly explain how I evaluate the performance of the UC-DR learner with that of the baseline double-robust estimator (DR) and with its undersampled version (U-DR). First, I compute the average bias of the estimator. This measure will give me an idea of how close the estimator is to the true value of the average treatment effect. Second, I compute the root mean squared error (RMSE) of the estimator. This measure, combined with the bias, will tell me how much the estimator varies across different samples. 

Finally, I use the different estimators in what is commonly called a "policy learning" problem. The idea is that I want to optimally assign the treatment to new observations. This means that, for a group of new individuals, I have to decide who should be treated, taking into account that the treatment has a certain cost. For simplicity, I assume that treating an individual bears a constant const equal to one. If the potential outcomes of each individual would be known, it would be easy to compute the optimal assignment. I would simply assign the treatment to the individuals where the effect of treatment $Y(1)-Y(0)$ exceeds the cost. However, in practice, the potential outcomes are not known. Athey and Wager[^5] (2021) show that it is possible to solve this problem by training a classification model using the $\hat\tau_i$ obtained from a doubly-robust estimator. I skip here all the details of this procedure, but you can find more information either directly in their paper or in Micheal Knaus' slides available in [his GitHub repo](https://github.com/MCKnaus/causalML-teaching). I will report the regret of the different estimators. The regret is defined as the difference between the average outcome of the optimal assignment and the average outcome of the assignment obtained by using the estimated $\hat\tau_i$ for a new random sample of 10'000 individuals. An estimator will perform well in this task if it is able to capture the heterogeneity in the treatment effect.

All the results are summarized in the table below. They show that:
1. **U-DR and UC-DR perform similarly in small samples:**
In small sample sizes with highly unbalanced treatment assignment, the U-DR and UC-DR estimators exhibit comparable performance in estimating causal effects. The UC-DR estimator has, however, a larger variance. This can be explained by the fact that an additional parameter has to be estimated ($\hat\gamma$).

2. **UC-DR excels with increased observations and decreasing unbalancedness:**
As the number of observations increases, the UC-DR estimator outperforms the U-DR estimator. Thanks to the adjustment, the UC-DR estimator can use more data to estimate the ATE and to train the policy learning classifier. This leads to a lower bias and a more accurate treatment assignment. However, the additional data used by the UC-DR estimator cannot offset the variance related with the estimation of $\hat\gamma$. In fact, the U-DR estimator still has the lowest RMSE.

3. **Limitations of baseline DR estimator in highly unbalanced samples:**
The baseline DR estimator, although widely used, faces challenges when the sample is highly unbalanced. In such cases, the propensity score predictions approach zero for some observations, leading to considerable outliers and severely biased causal effect estimates. This can result in misleading conclusions and limit the applicability of the baseline DR estimator in practical situations involving imbalanced datasets. Only for the largest and least unbalanced sample, the baseline DR estimator performs well.

    
|  | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; DR | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; U-DR | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; UC-DR |
| :----------------- | --------------------------: | --------------------------: | --------------------------: |
|_2% treated, N = 1'000, 1'000 simulations_ |
| Bias | 829'573.92  | **0.1321**    | 0.1346 |
| RMSE    | 25'756'549.85      | **0.5333**      | 0.5978 |
| Regret    | 0.6229      | **0.5804**      |0.5849 |
|_2% treated, N = 10'000, 100 simulations_ |
| Bias | 607'085.43  | 0.0769      | **0.03963** |
| RMSE    | 4'114'752.21      | **0.12253**      | 0.15762 |
| Regret    | 0.4466   | 0.4353   | **0.3865** |
|_12% treated, N = 1'000, 1'000 simulations_ |
| Bias | 61'249.58  | 0.1003   | **0.0533** |
| RMSE    | 1'272'147.90      | **0.1704**      | 0.2151 |
| Regret    | 0.4822      | 0.46856      | **0.44722** |
|_12% treated, N = 10'000, 100 simulations_ |
| Bias | **0.0124**  | 0.0613      | 0.0147 |
| RMSE    | **0.0371**      | 0.0491      | 0.0578 |
| Regret    | 0.3790      | 0.4039      | **0.3752** |


## Conclusion

So which estimator should we use in practice when dealing with unbalanced treatment assignment? As in many cases: it depends. If the sample is small and highly unbalanced, undersampling is a good choice. If the sample is large I would recommend to undersample only the data used to train the machine learning models and then adjust the propensity score predictions. This will allow to use more data to estimate the ATE or to train other models on the $\hat\tau_i$. This could be particularly useful when we are interested in conditional treatment effects or in solving a policy learning problem. In any case, the baseline doubly-robust estimator should be avoided in highly unbalanced samples, unless we have a very large sample size.


## *Proof of the asymptotic properties*

  *The proof of the asymptotic properties of the UC-DR estimator is very similar to the proof of the classical doubly-robust estimator as outlined in Stefan Wager's script[^1]. In the following I will therefore only cover the differences in the proofs to try to keep this part as short as possible.*
 
  *Notice that the estimator of $\gamma$ is simply a maximum likelihood estimator for which we have that $|\hat\gamma - \gamma|= o_p(n^{-1/2})$. 
  Moreover, I assume that $\epsilon < \gamma < 1 - \epsilon$ for $\epsilon>0$.*

  *I will assume that the previously mentioned conditions hold for the undersampled machine learner:*
  - *Overlap: $\eta < e_S(x) < 1-\eta$ for $\eta>0$ and for all $x \in \mathcal{X}$.*
  - *Consistency:*
  
  $$
    \sup_{x \in \mathcal{X}} |\hat{e}_S(x) - e_S(x)| \xrightarrow{p} 0
  $$
  
  - *Risk decay:*
  
  $$
    E\left[\left(\hat{\mu}_d(X) - \mu_d(X)\right)^2\right] E\left[\left(\hat{e}_S(x) - e_S(X)\right)^2\right] = o(n^{-1}).
  $$
  
  *Now by noticing that*
  
  $$
  \begin{aligned}
  |\hat{e}_S(x)\hat\gamma - e_S(X)\gamma| &= |\hat{e}_S(x)\hat\gamma - e_S(X)\hat\gamma + e_S(X)\hat\gamma - e_S(X)\gamma|\\ 
  & \leq |\hat{e}_S(x) - e_S(X)| |\hat\gamma| + |\hat\gamma - \gamma| |e_S(X)|\\ &\leq |\hat{e}_S(x) - e_S(X)| + |\hat\gamma - \gamma| 
  \end{aligned}
  $$
  
  *we can conclude that $\hat{e}_S(x)\hat\gamma$ is sup-norm consistent and therefore, thanks to the overlap assumption, $\hat{e}(X)$ is also sup-norm consistent.*
  
  *From here I follow the proof in Wager's script. I will focus on an estimator for $\theta_1=E[Y(1)]$.* 
  *Extending this proof to the ATE estimator is straight forward since $\theta=\theta_1 - \theta_0$.* 
  *First, if we would know the true functions $\mu_1(X)$ and $e(X)$, the oracle estimator:*
  
  $$
  \widetilde\theta_1 = \frac{1}{N} \sum_{i=1}^N \left(\mu_1(X_i) + \frac{D_i(Y_i-\mu_1(X_i))}{e(X_i)}\right)
  $$
  
  *would simply be an average of independent random variables and by the central limit theorem we would have that $\sqrt{N}(\widetilde\theta_1 - \theta_1) \xrightarrow{d} \mathcal{N}(0, V)$.*
  *Next, if we can show that $\sqrt{N}(\widetilde\theta_1 - \hat\theta_1)=o_p(1)$, we can conclude that our estimator converges to the same distribution as the oracle estimator.*
  
  *Since I use cross-fitting for the estimation, I can rewrite the estimator as follows:*
  
  $$
  \hat\theta_1 = \frac{|\mathcal{I}_1|}{N} \hat\theta_1^{\mathcal{I}_1} + \frac{|\mathcal{I}_2|}{N} \hat\theta_1^{\mathcal{I}_2}, \qquad \hat\theta_1^{\mathcal{I}_1} = \frac{1}{|\mathcal{I}_1|} \sum_{i \in \mathcal{I}_1} \left(\hat{\mu}_1^{\mathcal{I}_2^S}(X_i) + \frac{D_i(Y_i-\hat{\mu}^{\mathcal{I}_2^S}_1(X_i))}{\hat{e}^{\mathcal{I}_2^S}(X_i)}\right).
  $$
  
  *So it is sufficient to show that $\sqrt{N}(\widetilde\theta_1^{\mathcal{I}_1} - \hat\theta_1^{\mathcal{I}_1})=o_p(1)$.*
  *Stefan wager shows how we can decompose the difference into three terms:*
  
  $$
  \begin{aligned}
  \widetilde\theta_1^{\mathcal{I}_1} - \hat\theta_1^{\mathcal{I}_1} &= \frac{1}{|\mathcal{I}_1|} \sum_{i \in \mathcal{I}_1} \left( \left(\hat{\mu}_1^{\mathcal{I}_2^S}(X_i) - \mu_1(X_i)\right) \left(1-\frac{D_i}{e(X_i)}\right) \right)\qquad \text{(A)}\\ 
  &+ \frac{1}{|\mathcal{I}_1|} \sum_{i \in \mathcal{I}_1} D_i \left( \left( Y_i - \mu_1(X_i) \right) \left(\frac{1}{\hat{e}^{\mathcal{I}_2^S}(X_i)}-\frac{1}{e(X_i)}\right) \right) \qquad \text{(B)}\\ 
  &- \frac{1}{|\mathcal{I}_1|} \sum_{i \in \mathcal{I}_1} D_i \left( \left(\hat{\mu}_1^{\mathcal{I}_2^S}(X_i) - \mu_1(X_i)\right) \left(\frac{1}{\hat{e}^{\mathcal{I}_2^S}(X_i)}-\frac{1}{e(X_i)}\right) \right) \qquad \text{(C)}.
  \end{aligned}
  $$
  
  *We therefore have to show that each of these three components converge to zero at rate $N^{-1/2}$. First, (A) does not dependent on the estimation of the propensity score, and we can therefore use the same argument as in Stefan Wager's script, which is why I will skip this part of the proof. Second, I compute the squared $L_2$-norm of (B):*
  
  $$
  \begin{aligned}
  &\left\|\frac{1}{|\mathcal{I}_1|} \sum_{i \in \mathcal{I}_1} D_i \left( \left( Y_i - \mu_1(X_i) \right) \left(\frac{1}{\hat{e}^{\mathcal{I}_2^S}(X_i)}-\frac{1}{e(X_i)}\right) \right)\right\|_2^2\\ 
  &= E\left[ \left(\frac{1}{|\mathcal{I}_1|} \sum_{i \in \mathcal{I}_1} D_i  \left( Y_i - \mu_1(X_i) \right) \left(\frac{1}{\hat{e}^{\mathcal{I}_2^S}(X_i)}-\frac{1}{e(X_i)}\right) \right)^2 \right]\\
  &= E\left[ E\left[  \left(\frac{1}{|\mathcal{I}_1|} \sum_{i \in \mathcal{I}_1} D_i  \left( Y_i - \mu_1(X_i) \right) \left(\frac{1}{\hat{e}^{\mathcal{I}_2^S}(X_i)}-\frac{1}{e(X_i)}\right) \right)^2 \Bigg|\mathcal{I}_2^S, D_1, \dots, D_N  \right] \right]\\
  &= E\left[ Var\left[  \frac{1}{|\mathcal{I}_1|} \sum_{i \in \mathcal{I}_1} D_i  \left( Y_i - \mu_1(X_i) \right) \left(\frac{1}{\hat{e}^{\mathcal{I}_2^S}(X_i)}-\frac{1}{e(X_i)}\right)  \Bigg|\mathcal{I}_2^S, D_1, \dots, D_N  \right] \right]\\
  &= \frac{1}{|\mathcal{I}_1|^2} E\left[  \sum_{i \in \mathcal{I}_1} Var\left[   D_i  \left( Y_i - \mu_1(X_i) \right) \left(\frac{1}{\hat{e}^{\mathcal{I}_2^S}(X_i)}-\frac{1}{e(X_i)}\right)  \Bigg|\mathcal{I}_2^S, D_1, \dots, D_N  \right] \right]\\
  &= \frac{1}{|\mathcal{I}_1|} E\left[ Var\left[   D_i  \left( Y_i - \mu_1(X_i) \right) \left(\frac{1}{\hat{e}^{\mathcal{I}_2^S}(X_i)}-\frac{1}{e(X_i)}\right)  \Bigg|\mathcal{I}_2^S, D_1, \dots, D_N  \right] \right]\\
  &= \frac{1}{|\mathcal{I}_1|} E\left[ D_i  \left( Y_i - \mu_1(X_i) \right)^2 \left(\frac{1}{\hat{e}^{\mathcal{I}_2^S}(X_i)}-\frac{1}{e(X_i)}\right)^2 \right]\\
  &=\dots= \frac{o(1)}{N}
  \end{aligned}
  $$
  
  *I skipped the last steps since, thanks to the sup-consistency of the calibrated propensity score estimator, they coincide with the proof of the double-robust estimator. The main difference to the usual proof is the fact that I had to condition not only on the (undersampled) estimation sample, but also on the treatment assignments of the entire sample $\{D_i\}_{i=1,\dots, N}$. This step is necessary, since the calibrated propensity score depends on $\hat\gamma$ which is estimated over the entire sample. Despite this, the elements in the sum are still uncorrelated (forth equality):*
  
$$
  \begin{aligned}
  Cov\Bigg[& D_i \left( Y_i - \mu_1(X_i) \right) \left(\frac{1}{\hat{e}^{\mathcal{I}_2^S}(X_i)}-\frac{1}{e(X_i)}\right), D_j  \left( Y_j - \mu_1(X_j) \right) \left(\frac{1}{\hat{e}^{\mathcal{I}_2^S}(X_j)}-\frac{1}{e(X_j)}\right) \Bigg|\mathcal{I}_2^S, D_1, \dots, D_N \Bigg] \\
  = E\Bigg[& D_i  \left( Y_i - \mu_1(X_i) \right) \left(\frac{1}{\hat{e}^{\mathcal{I}_2^S}(X_i)}-\frac{1}{e(X_i)}\right) D_j  \left( Y_j - \mu_1(X_j) \right) \left(\frac{1}{\hat{e}^{\mathcal{I}_2^S}(X_j)}-\frac{1}{e(X_j)}\right) \Bigg|\mathcal{I}_2^S, D_1, \dots, D_N \Bigg] \\
  = E\Bigg[& D_i \left(\frac{1}{\hat{e}^{\mathcal{I}_2^S}(X_i)}-\frac{1}{e(X_i)}\right) D_j   \left(\frac{1}{\hat{e}^{\mathcal{I}_2^S}(X_j)}-\frac{1}{e(X_j)}\right)\\ 
   & \cdot E\left[ \left( Y_i - \mu_1(X_i) \right) \Big|X_i, D_i \right] E\left[ \left( Y_j - \mu_1(X_j) \right) \Big|X_j, D_j \right] \Bigg|\mathcal{I}_2^S, D_1, \dots, D_N \Bigg] = 0
  \end{aligned}
  $$
  
  
  *by the law of iterated expectations and the fact that the observations are independent.
  Lastly, we can focus on the (C):*
  
  $$
  \begin{aligned}
  & E\left[ \left( \hat{\mu}_1^{\mathcal{I}_2^S}(X_i) - \mu_1(X_i) \right) \left(\frac{1}{\hat{e}^{\mathcal{I}_2^S}(X_i)}-\frac{1}{e(X_i)}\right) \right]\\[0.2cm]
  & \leq E\left[ \left| \left( \hat{\mu}_1^{\mathcal{I}_2^S}(X_i) - \mu_1(X_i) \right) \left(\frac{1}{\hat{e}^{\mathcal{I}_2^S}(X_i)}-\frac{1}{e(X_i)}\right) \right| \right] \\[0.2cm]
  & = E\left[ \left| \left( \hat{\mu}_1^{\mathcal{I}_2^S}(X_i) - \mu_1(X_i) \right) \left(\frac{1}{\gamma e_S(X_i)} - \frac{1}{\hat\gamma \hat{e}^{\mathcal{I}_2^S}_S(X_i)} -\frac{1}{\gamma}+\frac{1}{\hat\gamma}\right) \right| \right]\\[0.2cm]
  & \leq E\left[ \Bigg| \hat{\mu}_1^{\mathcal{I}_2^S}(X_i) - \mu_1(X_i) \Bigg| \left| \left(\frac{1}{\gamma e_S(X_i)} - \frac{1}{\hat\gamma \hat{e}^{\mathcal{I}_2^S}_S(X_i)}\right) + \left(\frac{1}{\hat\gamma}-\frac{1}{\gamma} \right)\right| \right] \\[0.2cm]
  & \leq E\left[ \Bigg| \hat{\mu}_1^{\mathcal{I}_2^S}(X_i) - \mu_1(X_i) \Bigg| \left|\frac{1}{\gamma e_S(X_i)} - \frac{1}{\hat\gamma \hat{e}^{\mathcal{I}_2^S}_S(X_i)}\right| \right] + E\left[ \Bigg| \hat{\mu}_1^{\mathcal{I}_2^S}(X_i) - \mu_1(X_i) \Bigg| \left|\frac{1}{\hat\gamma}-\frac{1}{\gamma}\right| \right] \\[0.2cm]
  & \leq c_1 E\left[ \left| \hat{\mu}_1^{\mathcal{I}_2^S}(X_i) - \mu_1(X_i) \right| \left| \hat\gamma \hat{e}^{\mathcal{I}_2^S}_S(X_i) - \gamma e_S(X_i)\right| \right] + c_2 E\left[ \left| \hat{\mu}_1^{\mathcal{I}_2^S}(X_i) - \mu_1(X_i) \right| \left|\gamma-\hat\gamma\right| \right] \\[0.2cm]
    & = c_1 E\left[ \left| \hat{\mu}_1^{\mathcal{I}_2^S}(X_i) - \mu_1(X_i) \right| \left| \hat{e}^{\mathcal{I}_2^S}_S(X_i) - e_S(X_i)\right| |\hat\gamma | \right] \\[0.2cm]
    & \qquad + c_1 E\left[ \left| \hat{\mu}_1^{\mathcal{I}_2^S}(X_i) - \mu_1(X_i) \right| \left| \hat\gamma - \gamma \right| |e_S(X_i)| \right] \\[0.2cm]
    & \qquad + c_2 E\left[ \left| \hat{\mu}_1^{\mathcal{I}_2^S}(X_i) - \mu_1(X_i) \right| \left|\gamma-\hat\gamma\right| \right]\\[0.2cm]
    & \leq c_1 E\left[ \left| \hat{\mu}_1^{\mathcal{I}_2^S}(X_i) - \mu_1(X_i) \right| \left| \hat{e}^{\mathcal{I}_2^S}_S(X_i) - e_S(X_i)\right| \right]\\[0.2cm]
    & \qquad + c_1 E\left[ \left| \hat{\mu}_1^{\mathcal{I}_2^S}(X_i) - \mu_1(X_i) \right| \left| \hat\gamma - \gamma \right|\right]\\[0.2cm]
    & \qquad + c_2 E\left[ \left| \hat{\mu}_1^{\mathcal{I}_2^S}(X_i) - \mu_1(X_i) \right| \left|\gamma-\hat\gamma\right| \right] \\[0.2cm]
    & \leq c_1 \|\hat{\mu}_1^{\mathcal{I}_2^S}(X_i) - \mu_1(X_i) \|_2 \|\hat{e}^{\mathcal{I}_2^S}_S(X_i) - e_S(X_i) \|_2 + (c_1+c_2) \|\hat{\mu}_1^{\mathcal{I}_2^S}(X_i) - \mu_1(X_i) \|_2 \|\hat\gamma -\gamma \|_2\\[0.2cm]
    & = o(N^{-1/2}) + o(N^{-1/2}) = o(N^{-1/2})	
    \end{aligned}
  $$
  
  *where the first the first term in the last step is $o(N^{-1/2})$ by the risk-decay assumption and the second term is $o(N^{-1/2})$ by the convergence rate of $\hat\gamma$ and sup-consistency of $\hat\mu_1$. The positive constants $c_1$ and $c_2$ come from the boundedness of $e_S$ and $\gamma$ (and that of their respective estimators). Combined with the law of large numbers, we can conclude that the term (C) is $o_p(N^{-1/2})$. This concludes the proof of the theorem as all three terms (A), (B) and (C) are $o_p(N^{-1/2})$.*

## References

[^1]: Stefan Wager (2020), STATS 361: Causal Inference, Retrieved from [Wager's webpage](https://web.stanford.edu/~swager/teaching.html).

[^2]: Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W. and Robins, J. (2018), Double/debiased machine learning for treatment and structural parameters. _The Econometrics Journal_, 21: C1-C68. [doi:10.1111/ectj.12097](https://doi.org/10.1111/ectj.12097).

[^3]: A. D. Pozzolo, O. Caelen, R. A. Johnson and G. Bontempi (2015), Calibrating Probability with Undersampling for Unbalanced Classification. _IEEE Symposium Series on Computational Intelligence_, pp. 159-166, [doi: 10.1109/SSCI.2015.33](https://doi.org/10.1109/SSCI.2015.33).

[^4]: G. Okasa (2022), Meta-Learners for Estimation of Causal Effects: Finite Sample Cross-Fit Performance. _arXiv working paper_, [arXiv:2201.12692v1](https://doi.org/10.48550/arXiv.2201.12692)

[^5]: Athey, S. and Wager, S. (2021), Policy Learning With Observational Data. Econometrica, 89: 133-161. [https://doi.org/10.3982/ECTA15732](https://doi.org/10.3982/ECTA15732)