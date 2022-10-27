# Softmax Regression

$$
\newcommand{\b}{\mathbf b}
\newcommand{\w}{\mathbf w}
\newcommand{\x}{\mathbf x}
\newcommand{\y}{\mathbf y}
\newcommand{\z}{\mathbf z}
\newcommand{\W}{\mathbf W}
\newcommand{\E}{\mathbb E}
\DeclareMathOperator*{\Var}{Var}
\newcommand{\abs}[1]{\left\lvert #1 \right\rvert}
\newcommand{\brackets}[1]{\left[ #1 \right]}
\DeclareMathOperator*{\argmin}{argmin}
\DeclareMathOperator*{\softmax}{softmax}
\newcommand{\given}{\,\vert\,}
\newcommand{\L}{\mathcal L}
\newcommand{\D}{\mathcal D}
$$

## Non-binary classification

- when doing binary classification, output must be $0$ or $1$, so activation is straightforward
- when we have more than 2 classes, just one output variable with activation and rounding is not viable strategy - need to have vector of outputs and represent targets similarly
  - can represent targets as **one-hot vectors**, i.e. each target is in the form $e_k = (0, ..., 0, 1, 0, ..., 0)$ (only $i^\text{th}$ entry is 1)
  - each entry of output vector takes its own regression, i.e. each $y_k$ is the activation of a $z_k$ which is $z_k = \w_k^T \x + b_k$ as in a one-output regression
  - so if $f$ is the activation function, then $y_k = f(z_k) = f(\w_k^T \x + b_k)$
  - we can vectorize further by using a weight matrix rather than weight vectors, so $\y = f(\W \x + \b)$ ($f$ applied element-wise)
- need to choose $f$ so that entries of $\y$ are nonnegative and add up to 1, otherwise prediction is difficult to interpret
- softmax activation achieves this
  - $\softmax(\z)^T = \frac{\ds 1}{\ds \sum_{k=1}^K e^{z_k}} (e^{z_1}, ..., e^{z_K})$
  - so $\ds y_k = \softmax(\z)_k = \frac{e^{z_k}}{\ds \sum_{m=1}^K e^{z_m}}$
  - normalized, so $\ds \sum_{k=1}^K y_k = 1$ and each entry has $0 < y_k < 1$
  - uses exponential, so larger $z_k$ means *much* larger $y_k$
