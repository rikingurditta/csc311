# Exam Review

$$
\newcommand{\b}{\mathbf b}
\newcommand{\t}{\mathbf t}
\newcommand{\w}{\mathbf w}
\newcommand{\x}{\mathbf x}
\newcommand{\y}{\mathbf y}
\newcommand{\z}{\mathbf z}
\newcommand{\W}{\mathbf W}
\newcommand{\E}{\mathbb E}
\newcommand{\btheta}{\boldsymbol \theta}
\DeclareMathOperator*{\Var}{Var} 
\newcommand{\abs}[1]{\left\lvert #1 \right\rvert}
\newcommand{\brackets}[1]{\left[ #1 \right]}
\DeclareMathOperator*{\argmin}{argmin}
\DeclareMathOperator*{\argmax}{argmax}
\DeclareMathOperator*{\softmax}{softmax}
\newcommand{\given}{\,\vert\,}
\newcommand{\L}{\mathcal L}
\newcommand{\D}{\mathcal D}
\newcommand{\ds}{\displaystyle}
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

- can model probability of event with unknown parameters, and learn those parameters to follow observed data $\D$
- likelihood function $\ell$ is probability of observing data with parameters, so $\ell(\btheta)$ is how well $\btheta$ predicts observed outcomes
- MLE (maximum likelihood estimator) is $\ds \hat \btheta_{ML} = \argmax_\btheta \ell(\btheta)$
- MAP (maximum a posteriori) is $\ds \hat \btheta_{MAP} = \argmax_\btheta p(\btheta \given \D) = \argmax_\btheta p(\btheta) p(\D \given \btheta)$
- Naïve Bayes assumption: features $x_i$ are conditionally independent given class $c$: $p(c, \x) = p(c) p(x_1 \given c) \cdots p(x_D \given c)$

## Gaussian discriminant analysis

Can 
