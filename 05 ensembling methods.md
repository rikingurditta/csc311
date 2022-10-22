# Ensembling Methods

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
$$

## Bagging motivation

- sample $m$ independent training sets from $p_\text{sample}$
- compute hte prediction $y_i$ using each training set
- compute the average prediction $\displaystyle y = \frac{1}{m} \sum_{i=1}^m y_i$
- how does this affect the three terms of the expected loss?
  - bias is unchanged
    - $\ds \E[y] = \E\brackets{\frac{1}{m} \sum_{i=1}^m y_i} = \frac{1}{m}\sum_{i=1}^m \E[y_i] = \E[y_i]$
  - variance is reduced
    - $\ds \Var[y] = \Var\brackets{\frac{1}{m} \sum_{i=1}^m y_i} = \frac{1}{m^2} \sum_{i=1}^m \Var[y_i] = \frac{1}{m} \Var[y_i]$
  - bayes error is unchanged
    - we have no control over it

## Bagging

- $p_\text{sample}$ is usually expensive to sample from, so training separate models is wasteful
- **bootstrap aggregation**, aka **bagging**, is taking a training set $\D$ and using the empirical distribution $p_\D$ as a proxy for $p_\text{sample}$
  - suppose $\D$ has $n$ samples
  - make $m$ different data sets $\D_1, ..., \D_m$ each with $n$ samples taken from $\D$ with replacement
    - sampling with replacement so that they represent the underlying population that the original dataset is from, rather than depending on the original dataset
  - train a model on each $\D_i$
  - final prediction is average of predictions from $\D_i$'s'
- can be stronger than average model
- do not get $1/m$ variance if datasets are not independent
- can reduce correlation between datasets by introducing more variability
  - could use different algorithms
- weighing each model equally may not be best
  - weighted ensembling often better when the member models are very different


### Random forests

- random forest is bagging decision trees
  - could add more variability by allowing each tree to only split on certain features
  - decide these features randomly
