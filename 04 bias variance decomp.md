# Bias-Variance Decomposition

$$
\newcommand{\x}{\mathbf x}
\newcommand{\E}{\mathbb E}
\DeclareMathOperator*{\Var}{Var}
\newcommand{\abs}[1]{\left\lvert #1 \right\rvert}
\newcommand{\brackets}[1]{\left[ #1 \right]}
\DeclareMathOperator*{\argmin}{argmin}
\newcommand{\given}{\,\vert\,}
\newcommand{\L}{\mathcal L}
$$

## $\text{Var}[x]$ lemma

$$
\begin{align*}
\Var[x] &= \E[(x - \E[x])^2] \\
&= \E[x^2] - 2\E[x]\E[x] + \E[x]^2 \\
&= \E[x^2] - \E[x]^2
\end{align*}
$$

## Investigating squared error loss

Given an input vector $\x$ and the conditional distribution $p(t \given \x)$

Let $y_\ast = \E[t \given \x]$, then $y_\ast$ is the best possible prediction to minimize squared error loss $\L(y, t) = (y - t)^2$. We can show this by treating $y$ as a constant and $t$ as a random variable, and invetsigating the expected loss:
$$
\begin{align*}
\E[\L(y, t) \given \x] &= \E[(y - t)^2 \given \x] \\
&= \E[y^2 - 2yt + t^2 \given \x] \\
&= y^2 - 2y \E[t \given \x] + \E[t^2 \given \x] \\
&= y^2 - 2y \E[t \given \x] + E[t \given \x]^2 + \Var[t \given \x] \\
&= y^2 - 2yy_* + y_*^2 + \Var[t \given \x] \\
&= (y - y_*)^2 + \Var[t \given \x]
\end{align*}
$$
The first term can be minimized to 0 if we predict $y_*$, but the second term does not depend on $y$ so it cannot be optimized. We call $\Var[t \given \x]$ the **Bayes error**. Thus the best possible choice of $y$ is $y_*$, and if an algorithm predicts $y_*$ we call it **Bayes-optimal**.

Now let's treat $y$ as a random variable that depends on the dataset and $y_\ast$ as a constant:
$$
\begin{align*}
\E[(y - y_*)^2] &= y_*^2 - 2y_* \E[y] + \E[y^2] \\
\E[(y - y_*)^2] &= y_*^2 - 2y_* \E[y] + \E[y]^2 + \Var[y] \\
&= (y_* - \E[y])^2 + \Var[y]
\end{align*}
$$
Combining this with our expected loss, we get
$$
\E[\L(y, t) \given \x] = \underbrace{(y_* - \E[y \given \x])^2}_\text{bias} + \underbrace{\Var[y \given \x]}_\text{variance} + \underbrace{\Var[t \given \x]}_\text{Bayes error}
$$

## Interpreting bias-variance decomposition

- bias is how far the centre of our predictions is from the true predictions
  - i.e. low bias = high accuracy
  - corresponds to underfitting - an underfit model is not expected to predict well
- variance is how spread apart our predictions are
  - i.e. low variance = high precision
  - corresponds to overfitting - an overfit model might make wildly different predictions for small changes in input
- bayes error is how well we can predict the target given the data,
  - cannot optimize this within our learning algorithm, inherent to the data

## Choice of model

An overly simple model might have

- high bias
  - cannot capture structure in data
- low variance
  - enough data to get stable estimates

e.g. kNN with very large $k$ will tend to output an average of most points in the data set - does not depend very much on the specific input, and doesn't vary a lot because it always averages most points

An overly complex model might have

- low bias
  - captures all of the structure in the data (overfits)
- high variance
  - reflects all the quirks in the sampled data
