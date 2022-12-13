# Exam Review

$$
\newcommand{\b}{\mathbf b}
\newcommand{\t}{\mathbf t}
\newcommand{\w}{\mathbf w}
\newcommand{\x}{\mathbf x}
\newcommand{\y}{\mathbf y}
\newcommand{\z}{\mathbf z}
\newcommand{\U}{\mathbf U}
\newcommand{\W}{\mathbf W}
\newcommand{\E}{\mathbb E}
\newcommand{\L}{\mathcal L}
\newcommand{\J}{\mathcal J}
\newcommand{\D}{\mathcal D}
\newcommand{\btheta}{\boldsymbol \theta}
\newcommand{\bmu}{\boldsymbol \mu}
\newcommand{\bSigma}{\boldsymbol \Sigma}

\DeclareMathOperator*{\Var}{Var}
\DeclareMathOperator*{\argmin}{argmin}
\DeclareMathOperator*{\argmax}{argmax}
\DeclareMathOperator*{\softmax}{softmax}

\newcommand{\abs}[1]{\left\lvert #1 \right\rvert}
\newcommand{\parens}[1]{\left( #1 \right)}
\newcommand{\brackets}[1]{\left[ #1 \right]}
\newcommand{\partials}[2]{\frac{\partial #1}{\partial #2}}
\newcommand{\given}{\,\vert\,}

\newcommand{\ds}{\displaystyle}
$$

## Bayes rule

$$
p(x \given y) = \frac{p(y \given x) p(x)}{p(y)}
$$

## Types of machine learning

- supervised: data has labels
  - classification
- unsupervised: data has no labels
  - clustering
- reinforcement: interacts with the world to develop policy that maximizes reward

## Bias-variance decomposition

$$
\E[(y - t)^2] = \underset{\text{bias}}{(y_\ast - \E[y])^2} + \underset{\text{variance}}{\Var[y]} + \underset{\text{Bayes error}}{\Var[t]}
$$

$y_\ast = \E[t \given \x]$, i.e. the best possible prediction we can make - if an algorithm achieves this then it is called Bayes optimal

- bias is accuracy
- variance is how spread out predictions are
- Bayes error is inherent to training data, cannot be optimized
- what this decomposition tells us:
  - underfit model
    - high bias - inaccurate
    - low variance - stable

  - overfit model
    - low bias - fit very well
    - high variance - may change a lot for small changes in input since it captures more quirks/details of training data


## Entropy

$$
\begin{align*}
\text{entropy}:
H(X) &= -\sum_{x \in X} p(x) \log_2 p(x) \\
\text{joint entropy}:
H(X, Y) &= -\sum_{x \in X} \sum_{y \in Y} p(x, y) \log_2 p(x, y) \\
\text{conditional entropy}:
H(X \given Y = y) &= -\sum_{x \in X} p(x \given Y = y) \log_2 p(x \given Y = y) \\
\text{expected conditional entropy}:
H(X \given Y) &= \E_{y \in Y}[H(X \given Y = y)] \\
&= -\sum_{x \in X} \sum_{y \in Y} p(x, y) \log_2p(x \given y) \\
\text{chain rule for entropy}:
H(X, Y) &= H(X) + H(Y \given X) = H(Y) + H(X \given Y) \\
\text{information gain}:
IG(X \given Y) &= H(X) - H(X \given Y)
\end{align*}
$$

## Bagging

- if $\D$ is original dataset, create multiple proxy datasets each of size $\abs{\D}$ by sampling with replacement
- train a model on each dataset
- use vote/average to combine each model's predictions into one
- bias is unchanged
- variance is decreased

## Generalized linear models

- generalized linear models all have linear decision boundaries

### Linear regression

- model is $y = \w^T \x + b$
  - can rewrite with $b$ appended to $\w$, and assuming last entry of $\x$ is $1$
- per-item loss function is $\L(y, t) = (y - t)^2 / 2$
- average loss function is $\ds \J(\w) = \frac{1}{N} \sum_i^N (y^{(i)} - t^{(i)})^2 / 2 = \frac{1}{N} \sum_i^N (\w^T \x^{(i)} - t^{(i)})^2 / 2$
- regularization reduces overfitting by penalizing large $\w$ - $\J_{reg}(\w) = \J(\w) + \w^T \w / 2$
- can train by descending on negative gradient of loss
  - $\ds \nabla_\w \J(w) = \frac{1}{N} \sum_i^N (y^{(i)} - t^{(i)}) \x^{(i)}$
  - $\ds \nabla_\w \J_{reg}(w) = \w + \frac{1}{N} \sum_i^N (y^{(i)} - t^{(i)}) \x^{(i)}$

### Logistic regression

- for binary classification, linear regression must be rounded somehow
- $\ds \sigma(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{1 + e^x}$ is sigmoid function, $\sigma(\R) = (0, 1)$
- this is smooth transition from 0 to 1 as $x$ goes from negative to positive, so good for binary classification
- new model can be $y = \sigma(\w^T \x)$
  - implicit constant bias is included
- $\sigma$ has very small derivative for very large $\abs x$, so gradient of $\sigma(\w^T \x)$ would be small when $\w^T \x$ is large, which hampers training - need to choose loss which doesn't use gradient of $\sigma$
- cross-entropy $\L_{CE}$ is good for this - $\L_{CE}(y, t) = -t\log(y) - (1 - t)\log(1 - y)$
  - $\ds \nabla_\w \L_{CE}(y = \w^T\x, t) = (y - t) \x$
  - thiis gradient of loss is the same as for linear regression with least squares loss, so update step is the same
- decision boundary is linear

### Softmax regression

- generalizes logistic regression to $K$ classes
- $\W \in \R^{K \times (D + 1)}$
  - $D + 1$ since $\x$ has a $1$ appended to the end, for bias term
- $\z = \W \x$
- $\y = \ds \softmax(\z) = \sum_k^K \frac{1}{e^{z_k}}(e^{z_1}, ..., e^{z_K})$
- use cross entropy loss $\ds \L_{CE}(\y, \t) = -\t^T \log(\y)$
- resulting update step is same as for logistic for each row of $\w$

### Linear separability

- binary classification can be done with a generalized linear model when the underlying data is linearly separable
- linearly separable data is convex, i.e. for a class $C$, if $\x, \y \in C$ then $p = (1 - t)\x + t\y \in C$ for every $t$

## Descent methods

- gradient descent can be slow because we are taking each step based on every data point
- can also use stochastic gradient descent (SGD), taking steps based on single data points
- can also use batched gradient descent, taking steps based on subsets of dataset

## Neural networks

- hypothesis space is space of functions that can be learned by neural networks
  - closed under composition - layers can be composed
  - closed under linear combination - layers can be added
- more layers make the network more expressive
  - this is not necessarily good, as too expressive of a network could be able to fit multiple functions to the same dataset
  - can prevent overfitting with $L^2$ regularization or with early stopping

### Error signals

For loss function $\L$, we define the error signal $\ds \overline x =  \partials{\L}{x}$

- if $\y$ and $\z$ depend on $\x$ in the computation of $\L$ (and no other variables depend on $\x$, and $\y, \z$ do not depend on each other), then $\ds \overline \x = \partials{\L}{\x} = \partials{\L}{\y} \partials{\y}{\x} + \partials{\L}{\z} \partials{\z}{\x} = \overline \y \partials{\y}{\x} + \overline \z \partials{\z}{\x}$

## Probabilistic modelling

- can model probability of event with unknown parameters $\btheta$, and learn those parameters to follow observed data $\D$

- MLE (maximum likelihood estimator) is $\ds \hat \btheta_{ML} = \argmax_\btheta \ell(\btheta)$

  - likelihood function $\ell$ is probability of observing data with parameters, so $\ell(\btheta) = p(\D \given \btheta)$ is how well $\btheta$ predicts observed outcomes

- MAP (maximum a posteriori) is $\ds \hat \btheta_{MAP} = \argmax_\btheta p(\btheta \given \D) = \argmax_\btheta p(\btheta) p(\D \given \btheta)$

  - $p(\btheta)$ is the prior, i.e. our beliefs about the data

  - posterior $p(\btheta \given \D)$ can be thought of as "updating our beliefs about the data using our observations"

  - $$
    p(\btheta \given \D) = \frac{p(\btheta) p(\D \given \btheta)}{p(\D)} = \frac{p(\btheta) p(\D \given \btheta)}{\int p(\boldsymbol \phi) p(\D \given \boldsymbol \phi) d\boldsymbol \phi}
    $$

  - denominator is intractable so we drop it

- Naïve Bayes assumption: features $x_i$ are conditionally independent given class $c$: $p(c, \x) = p(c) p(x_1 \given c) \cdots p(x_D \given c)$

### Discriminative vs generative classifiers

- discriminative classifiers try to learn class labels from the space of inputs

  - take features $\x$ and try to figure out class probability $p(y \given \x)$

- generative classifiers try to build a model of the data, i.e. $p(\x, y)$

  - if $p(y)$ is known then we can combine these to compute $p(\x \given y)$ directly

  - we can then use Bayes rule to compute $p(y \given \x)$

  - $$
    p(y \given \x) = \frac{p(\x \given y) p(y)}{p(\x)} = \frac{p(\x \given y) p(y)}{\sum_k p(\x \given y_k) p(y_k)}
    $$

## Gaussian discriminant analysis

- mean $\bmu = \E[\x] = (\mu_1, ..., \mu_D)$
- covariance $\bSigma = \E[(\x - \bmu) (\x - \bmu)^T] = \begin{pmatrix} \sigma_1^2 & \sigma_{12} & \cdots & \sigma_{1D} \\ \sigma_{21} & \sigma_2^2 & \cdots & \sigma_{2D} \\ \vdots & \vdots & \ddots & \vdots \\ \sigma_{D1} & \sigma_{D2} & \cdots & \sigma_D^2 \end{pmatrix}$
- multivariate Gaussian distribution $\ds N(\x; \bmu, \bSigma) = \frac{1}{\sqrt{(2\pi)^D \abs{\bSigma}}} \exp\parens{-\frac{1}{2} (\x - \bmu)^T \bSigma^{-1} (\x - \bmu)}$
- can use as a classifier model by finding a different multivariate Gaussian (different mean and covariance) for each class, i.e. $p(\x \given t_k) = N(\x; \bmu_k, \bSigma_k)$ for each $k$

## Matrix factorization

dimensionality reduction - projects high dimensional data in into lower dimensional (affine) subspace (or manifold!), i.e. from $\R^D$ to something that looks like $\R^K$ with $K \leq D$

### Principal component analysis

- want to find $\U, \hat\bmu$ so that we can project $\x$ to $\z$ which is in a lower dimensional space as $\z = \U^T (\x - \hat\bmu)$, and we can "reconstruct" $\x$ from $\z$ as $\tilde \x = \U \z + \hat\bmu$
  - want to minimize reconstruction error $\ds \frac{1}{N} \sum_i^N \abs{\x^{(i)} - \tilde\x^{(i)}}^2$
  - want to maximize reconstruction variability want to minimize reconstruction error $\ds \frac{1}{N} \sum_i^N \abs{\tilde\x^{(i)} - \hat\bmu}^2$
  - these are equivalent
- choose $\hat\bmu$ as the mean of the data $\x$
- choose $\U$ as the matrix whose columns are the $K$ eigenvectors of the covariance matrix $\bSigma$ with the greatest eigenvalues

### PCA for matrix factorization

- can use PCA to factor $\mathbf X = \mathbf Z \U^T$, i.e. project each row of $\mathbf X$ onto a lower dimensional subspace to get $\mathbf Z$ then reconstruct it using $\U^T$ (assuming $\mathbf X$ is centred)
- this is related to singular value decomposition

### PCA matrix completion using ALS

- if matrix has missing entries, can try to find optimal completion by minimizing $\ds \min_{\U, \mathbf Z} \frac{1}{2} (\mathbf X^T - \U \mathbf Z^T) = \min_{\U, \mathbf Z} \frac{1}{2} \sum_{(i, j)} (R_{ij} - \mathbf u_i^T \z_j)$
- can do this with alternating least squares - solve least squares problem for $\U$, then for $\mathbf Z$, repeat alternating solves until convergence

### Autoencoders

- a linear autoencoder will solve for the PCA
- a nonlinear autoencoder's latent space will be a nonlinear manifold rather than a subspace

## K-means

- cluster points by choosing $k$ centres and assigning points to appropriate centres, thus segmenting
- do by alternating optimization
  - assign points to centres
    - choose nearest centre
  - refit - choose better centres
    - choose mean of assigned points
- converges because always reduces cost
- may converge to a local minimum
