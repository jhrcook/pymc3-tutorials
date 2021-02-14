# LKJ Cholesky Covariance Priors for Multivariate Normal Models

> While the inverse-Wishart distribution is the conjugate prior for the covariance matrix of a multivariate normal distribution, it is not very well-suited to modern Bayesian computational methods.
> For this reason, the LKJ prior is recommended when modeling the covariance matrix of a multivariate normal distribution.

We will demonstrate modeling the covariance with the LKJ distribution using mock data.

```python
import warnings

import arviz as az
import numpy as np
import pandas as pd
import plotnine as gg
import pymc3 as pm
import seaborn as sns
from icecream import ic
from matplotlib import pyplot as plt

warnings.simplefilter(action="ignore", category=FutureWarning)
RANDOM_SEED = 8924
np.random.seed(3264602)
gg.theme_set(gg.theme_minimal())

%config InlineBackend.figure_format = "retina"
```

```python
N = 10000

μ_actual = np.array([1.0, -2.0])
σ_actual = np.array([0.7, 1.5])
Ρ_actual = np.matrix([[1.0, -0.4], [-0.4, 1.0]])

Σ_actual = np.diag(σ_actual) * Ρ_actual * np.diag(σ_actual)

x = np.random.multivariate_normal(μ_actual, Σ_actual, size=N)

ic(Σ_actual);
```

    ic| Σ_actual: matrix([[ 0.49, -0.42],
                          [-0.42,  2.25]])

```python
var, U = np.linalg.eig(Σ_actual)
angle = 180.0 / np.pi * np.arccos(np.abs(U[0, 0]))

plot_data = pd.DataFrame(x, columns=("x1", "x2"))

(
    gg.ggplot(plot_data, gg.aes(x="x1", y="x2"))
    + gg.geom_density_2d(alpha=0.5, size=1.3, color="blue")
    + gg.geom_point(alpha=0.05)
    + gg.geom_point(
        data=pd.DataFrame(μ_actual.reshape((1, 2)), columns=("x1", "x2")),
        color="red",
        shape="x",
        size=2,
    )
    + gg.labs(x="$x_1$", y="$x_2$", title="Mock data")
)
```

![png](024_lkj-cholesky-covariance-priors_files/024_lkj-cholesky-covariance-priors_3_0.png)

    <ggplot: (362124301)>

The sampling distribution for the multivariate normal model is $\textbf{x} \sim N(\mu, \Sigma)$ where $\Sigma$ is the covariance matrix: $\Sigma = \text{Cov}(x_i, x_j)$.

The LKJ distribution puts a prior on the correlation matrix, $\textbf{C} = \text{Corr}(x_i, x_j)$.
When combined with priors on the standard deviations on each component, this induces a prior on the covariance matrix $\Sigma$.

Inverting $\Sigma$ is unstable and slow, so we can use the [Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition), $\Sigma = \textbf{LL}^T$, where $\textbf{L}$ is a lower-triangular matrix.

PyMC3 supports the LKJ priors for the Cholesky decomposition of the covariance matrix via the `LKJCholeskyCov()` distribution.
It has two parameters: 1) `n` is the dimensions of the observations $\textbf{x}$ and 2) the PyMC2 distribution of the component standard deviations.
It also has a hyperparameter `eta` which controls the amount of correlation between components of $\textbf{x}$.
A value of `eta=1` creates a uniform distribution on correlations matrix, and the magnitude of correlations decreases as `eta` increases.

Usually, we are interested in the posteriors of the correlations matrix and the standard deviations, not the Cholesky covariance matrix (they are more interpretable and have scientific meaning in the model).
We can have PyMC3 automatically compute these posteriors and store them in the trace by setting `compute_corr=True`.

```python
with pm.Model() as model:
    chol, corr, stds = pm.LKJCholeskyCov(
        "chol", n=2, eta=2.0, sd_dist=pm.Exponential.dist(1.0), compute_corr=True
    )
    cov = pm.Deterministic("cov", chol.dot(chol.T))

    μ = pm.Normal("μ", 0.0, 1.5, shape=2, testval=x.mean(axis=0))
    obs = pm.MvNormal("obs", μ, chol=chol, observed=x)

    trace = pm.sample(init="adapt_diag", random_seed=RANDOM_SEED)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [μ, chol]

<div>
    <style>
        /*Turns off some styling*/
        progress {
            /*gets rid of default border in Firefox and Opera.*/
            border: none;
            /*Needs to be in here for Safari polyfill so background images work as expected.*/
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='4000' class='' max='4000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [4000/4000 01:05<00:00 Sampling 2 chains, 0 divergences]
</div>

    Sampling 2 chains for 1_000 tune and 1_000 draw iterations (2_000 + 2_000 draws total) took 75 seconds.

```python
az_model = az.from_pymc3(trace=trace, model=model)
az.summary(az_model, hdi_prob=0.89)
```

    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/arviz/stats/diagnostics.py:642: RuntimeWarning: invalid value encountered in double_scalars

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_5.5%</th>
      <th>hdi_94.5%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>μ[0]</th>
      <td>1.017</td>
      <td>0.007</td>
      <td>1.007</td>
      <td>1.028</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>2496.0</td>
      <td>2494.0</td>
      <td>2510.0</td>
      <td>1760.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>μ[1]</th>
      <td>-2.032</td>
      <td>0.015</td>
      <td>-2.060</td>
      <td>-2.011</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>2700.0</td>
      <td>2695.0</td>
      <td>2696.0</td>
      <td>1746.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>chol[0]</th>
      <td>0.694</td>
      <td>0.005</td>
      <td>0.686</td>
      <td>0.701</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>2625.0</td>
      <td>2625.0</td>
      <td>2631.0</td>
      <td>1533.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>chol[1]</th>
      <td>-0.589</td>
      <td>0.015</td>
      <td>-0.612</td>
      <td>-0.565</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>2646.0</td>
      <td>2646.0</td>
      <td>2636.0</td>
      <td>1568.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>chol[2]</th>
      <td>1.378</td>
      <td>0.009</td>
      <td>1.363</td>
      <td>1.393</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>3356.0</td>
      <td>3356.0</td>
      <td>3343.0</td>
      <td>1407.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>chol_stds[0]</th>
      <td>0.694</td>
      <td>0.005</td>
      <td>0.686</td>
      <td>0.701</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>2625.0</td>
      <td>2625.0</td>
      <td>2631.0</td>
      <td>1533.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>chol_stds[1]</th>
      <td>1.499</td>
      <td>0.011</td>
      <td>1.483</td>
      <td>1.517</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>3336.0</td>
      <td>3336.0</td>
      <td>3328.0</td>
      <td>1666.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>chol_corr[0,0]</th>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>2000.0</td>
      <td>2000.0</td>
      <td>2000.0</td>
      <td>2000.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>chol_corr[0,1]</th>
      <td>-0.393</td>
      <td>0.009</td>
      <td>-0.407</td>
      <td>-0.380</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>2641.0</td>
      <td>2641.0</td>
      <td>2648.0</td>
      <td>1518.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>chol_corr[1,0]</th>
      <td>-0.393</td>
      <td>0.009</td>
      <td>-0.407</td>
      <td>-0.380</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>2641.0</td>
      <td>2641.0</td>
      <td>2648.0</td>
      <td>1518.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>chol_corr[1,1]</th>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>2000.0</td>
      <td>2000.0</td>
      <td>1721.0</td>
      <td>1786.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>cov[0,0]</th>
      <td>0.482</td>
      <td>0.007</td>
      <td>0.471</td>
      <td>0.492</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>2630.0</td>
      <td>2630.0</td>
      <td>2631.0</td>
      <td>1533.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>cov[0,1]</th>
      <td>-0.409</td>
      <td>0.012</td>
      <td>-0.426</td>
      <td>-0.390</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>2399.0</td>
      <td>2399.0</td>
      <td>2409.0</td>
      <td>1438.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>cov[1,0]</th>
      <td>-0.409</td>
      <td>0.012</td>
      <td>-0.426</td>
      <td>-0.390</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>2399.0</td>
      <td>2399.0</td>
      <td>2409.0</td>
      <td>1438.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>cov[1,1]</th>
      <td>2.247</td>
      <td>0.032</td>
      <td>2.199</td>
      <td>2.300</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>3338.0</td>
      <td>3338.0</td>
      <td>3328.0</td>
      <td>1666.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
az.plot_trace(
    az_model,
    var_names=["~chol"],
    compact=True,
    lines=[
        ("μ", {}, μ_actual),
        ("cov", {}, Σ_actual),
        ("chol_stds", {}, σ_actual),
        ("chol_corr", {}, Ρ_actual),
    ],
);
```

    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/arviz/stats/density_utils.py:783: RuntimeWarning: divide by zero encountered in true_divide
    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/arviz/stats/density_utils.py:783: RuntimeWarning: invalid value encountered in true_divide
    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/arviz/stats/density_utils.py:783: RuntimeWarning: divide by zero encountered in true_divide
    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/arviz/stats/density_utils.py:783: RuntimeWarning: invalid value encountered in true_divide
    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/arviz/stats/density_utils.py:1037: RuntimeWarning: divide by zero encountered in true_divide
    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/arviz/stats/density_utils.py:1037: RuntimeWarning: invalid value encountered in true_divide
    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/arviz/stats/density_utils.py:760: RuntimeWarning: divide by zero encountered in true_divide
    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/arviz/stats/density_utils.py:760: RuntimeWarning: invalid value encountered in true_divide
    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/arviz/stats/density_utils.py:763: RuntimeWarning: divide by zero encountered in double_scalars
    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/arviz/stats/density_utils.py:770: UserWarning: Something failed when estimating the bandwidth. Please check your data
    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/arviz/stats/density_utils.py:1037: RuntimeWarning: divide by zero encountered in true_divide
    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/arviz/stats/density_utils.py:1037: RuntimeWarning: invalid value encountered in true_divide
    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/arviz/stats/density_utils.py:760: RuntimeWarning: divide by zero encountered in true_divide
    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/arviz/stats/density_utils.py:760: RuntimeWarning: invalid value encountered in true_divide
    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/arviz/stats/density_utils.py:763: RuntimeWarning: divide by zero encountered in double_scalars
    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/arviz/stats/density_utils.py:770: UserWarning: Something failed when estimating the bandwidth. Please check your data

![png](024_lkj-cholesky-covariance-priors_files/024_lkj-cholesky-covariance-priors_7_1.png)

Get the posterior predictions and compare to the actual values.

```python
μ_post = trace["μ"].mean(axis=0)
μ_post
```

    array([ 1.01671391, -2.03193021])

```python
# Error in μ
(1 - μ_post / μ_actual).round(2)
```

    array([-0.02, -0.02])

```python
Σ_post = trace["cov"].mean(axis=0)
Σ_post
```

    array([[ 0.48176829, -0.40905125],
           [-0.40905125,  2.24689291]])

```python
# Error in Σ
(1 - Σ_post / Σ_actual).round(2)
```

    array([[0.02, 0.03],
           [0.03, 0.  ]])

---

```python
%load_ext watermark
%watermark -n -u -v -iv -w
```

    Last updated: Sun Feb 14 2021

    Python implementation: CPython
    Python version       : 3.9.1
    IPython version      : 7.20.0

    pandas    : 1.2.2
    plotnine  : 0.7.1
    arviz     : 0.11.1
    pymc3     : 3.11.1
    seaborn   : 0.11.1
    numpy     : 1.20.1
    matplotlib: 3.3.4

    Watermark: 2.1.0

```python

```
