# Gradient Descent

$$
\newcommand{\x}{\mathbf x}
\newcommand{\E}{\mathbb E}
\DeclareMathOperator*{\Var}{Var}
\newcommand{\abs}[1]{\left\lvert #1 \right\rvert}
\newcommand{\brackets}[1]{\left[ #1 \right]}
\DeclareMathOperator*{\argmin}{argmin}
\newcommand{\given}{\,\vert\,}
\newcommand{\L}{\mathcal L}
\newcommand{\D}{\mathcal D}
\newcommand{\partials}[2]{\frac{\partial #1}{\partial #2}}
\newcommand{\ds}{\displaystyle}
$$

## Speeding up gradient descent

- Computing gradient for entire dataset is very expensive
  - $\ds \partials{\L}{\theta} = \frac{1}{N} \sum_{i=1}^N \partials{\L^{(i)}}{\theta}$
  - 