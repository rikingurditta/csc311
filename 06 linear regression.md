# Linear Regression

$$
\newcommand{\x}{\mathbf x}
\newcommand{\E}{\mathbb E}
\DeclareMathOperator*{\Var}{Var}
\newcommand{\abs}[1]{\left\lvert #1 \right\rvert}
\newcommand{\norm}[1]{\left\lVert #1 \right\rVert}
\newcommand{\brackets}[1]{\left[ #1 \right]}
\DeclareMathOperator*{\argmin}{argmin}
\newcommand{\given}{\,\vert\,}
\newcommand{\L}{\mathcal L}
\newcommand{\J}{\mathcal J}
\newcommand{\D}{\mathcal D}
\newcommand{\w}{\mathbf w}
$$

## Regularization

- size of feature mapping can control complexity of model
  - e.g. in polynomial regression, can take more and more terms 
- another way to control complexity to improve generalization is regularization - penalize weights for getting too big
- $\ell^2$ regularization - add squared $\ell^2$ norm of weights to cost function to penalize large weights
  - $\ds \J_{reg}(\w) = \J(\w) + \frac{1}{2} \norm{\w}_2^2 = \J(\w) + \frac{1}{2} \w \cdot \w$
- if using gradient descent (coming up), resulting gradient is $\ds \partials{\J_{reg}}{\w} = \partials{\J}{\w} + \w$
