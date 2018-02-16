# regularized_glm

A simple python package for fitting L2- and smoothing-penalized generalized linear models.

Built primarily because the statsmodels GLM `fit_regularized` method is built to do elastic net (combination of L1 and L2 penalities), but if you just want to do an L2 or a smoothing penalty (like in generalized additive models), using a penalized iteratively reweighted least squares (p-IRLS) is much faster.

### Installation

```bash
pip install regularized_glm
```

OR

```bash
conda install -c edeno regularized_glm
```

### Other packages
+ [Statsmodels](https://github.com/statsmodels/statsmodels)
+ [pyGAM](https://github.com/dswah/pyGAM)

### References
Wood, S. (2006). Generalized additive models: an introduction with R (CRC press).
