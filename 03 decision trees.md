# Decision Trees

$$
\newcommand{\x}{\mathbf x}
\newcommand{\abs}[1]{\left\lvert #1 \right\rvert}
\DeclareMathOperator*{\argmin}{argmin}
\newcommand{\given}{\,\vert\,}
$$

## Information gain

### Entropy

For a discrete random variable $X$,
$$
H(X) = -\sum_{x \in X} p(x) \log_2 p(x)
$$
For continuous $X$,
$$
H(X) = -\int_X p(x) \log_2 p(x) dx
$$

### Joint entropy

For $X$, $Y$ discrete, the joint entropy is
$$
H(X, Y) = -\sum_{x \in X} \sum_{y \in Y} p(x, y) \log_2 p(x, y)
$$

### Conditional entropy

For $X$ discrete,
$$
H(X \given Y = y) = -\sum_{x \in X} p(x \given Y = y) \log_2 p(x \given Y = y)
$$

### Expected conditional entropy

For $Y$ discrete,
$$
\begin{align*}
H(X \given Y) &= \E_{y \in Y}[H(X \given Y = y)] \\
&= \sum_{y \in Y} p(y) H(X \given Y=y) \\
&= -\sum_{x \in X} \sum_{y \in Y} p(x, y) \log_2p(x \given y)
\end{align*}
$$
