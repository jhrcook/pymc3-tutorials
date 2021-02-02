# Getting started with PyMC3

[Tutorial](https://docs.pymc.io/notebooks/getting_started.html)

```python
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import pymc3 as pm

%config InlineBackend.figure_format = "retina"
az.style.use("arviz-darkgrid")
gg.theme_set(gg.theme_minimal)
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
```

## A Motivating Example: Linear Regression

$$
Y \sim \mathcal{N}(\mu, \sigma^2) \\
\mu = \alpha + \beta_1 X_1 + \beta_2 X_2 \\
\alpha \sim \mathcal{N}(0, 10) \\
\beta_1 \sim \mathcal{N}(0, 10) \\
\beta_2 \sim \mathcal{N}(0, 10) \\
\sigma \sim |\mathcal{N}(0, 1)|
$$

```python
alpha, sigma = 1, 1
beta = [1, 2.5]

size = 100

X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

Y = alpha + beta[0] * X1 + beta[1] * X2 + (np.random.randn(size) * sigma)

d = pd.DataFrame({"X1": X1, "X2": X2, "Y": Y}).melt(id_vars="Y")

(
    gg.ggplot(d, gg.aes("value", "Y"))
    + gg.facet_wrap("variable", nrow=1, scales="fixed")
    + gg.geom_point()
)
```

    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/plotnine/facets/facet_wrap.py:215: UserWarning: This figure was using constrained_layout==True, but that is incompatible with subplots_adjust and or tight_layout: setting constrained_layout==False.

![png](001_getting-started-with-pymc3_files/001_getting-started-with-pymc3_3_1.png)

    <ggplot: (353867114)>

```python
with pm.Model() as basic_model:
    alpha = pm.Normal("alpha", 0, 10)
    beta = pm.Normal("beta", 0, 10, shape=2)
    sigma = pm.HalfNormal("sigma", 1)

    mu = alpha + beta[0] * X1 + beta[1] * X2

    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

    trace = pm.sample(1000, return_inferencedata=True)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [sigma, beta, alpha]

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
  100.00% [4000/4000 00:16<00:00 Sampling 2 chains, 0 divergences]
</div>

    Sampling 2 chains for 1_000 tune and 1_000 draw iterations (2_000 + 2_000 draws total) took 27 seconds.

```python
az.plot_trace(trace, compact=False)
plt.show()
```

![png](001_getting-started-with-pymc3_files/001_getting-started-with-pymc3_5_0.png)

```python
az.summary(trace)
```

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
      <th>hdi_3%</th>
      <th>hdi_97%</th>
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
      <th>alpha</th>
      <td>0.955</td>
      <td>0.108</td>
      <td>0.739</td>
      <td>1.144</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2722.0</td>
      <td>2620.0</td>
      <td>2757.0</td>
      <td>1306.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>beta[0]</th>
      <td>1.100</td>
      <td>0.114</td>
      <td>0.885</td>
      <td>1.311</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>2606.0</td>
      <td>2606.0</td>
      <td>2604.0</td>
      <td>1611.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>beta[1]</th>
      <td>2.952</td>
      <td>0.527</td>
      <td>1.916</td>
      <td>3.905</td>
      <td>0.009</td>
      <td>0.006</td>
      <td>3411.0</td>
      <td>3328.0</td>
      <td>3431.0</td>
      <td>1531.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma</th>
      <td>1.064</td>
      <td>0.077</td>
      <td>0.923</td>
      <td>1.207</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2722.0</td>
      <td>2715.0</td>
      <td>2780.0</td>
      <td>1433.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

---

## Case Study 1: Stochastic volatility

A case study on stock market volatility.
The distribution of market returns is non-normal, making smapling more difficult.

Asset prices have *time-varying volatility*, variance of day-over-day returns.
Sometimes, returns are highly volatile and other times prices are more stable.
This *stochasitc volatility model* addresses this with a latent volatility variable than changes over time.

$$
\log(r_i) \sim t(\nu, 0, \exp(-2s_i)) \\
s_i \sim \mathcal{N}(s_{i-1}, \sigma^2) \\
\nu \sim \exp(0.1) \\
\sigma \sim \exp(50)
$$

Here, $R$ is the faily return series modeled with a Student's *t*-distribution with an unknown degrees of freedom parameter $\nu$ and a scale parameter determined by a latent process $s$.
An individual $s_i$ is an individual daily log volatilities in the latent log volatility process.

Use data from S&P 500 indexsince the 2008 crisis.

```python
returns = pd.read_csv(
    pm.get_data("SP500.csv"), parse_dates=True, index_col=0, usecols=["Date", "change"]
).reset_index(drop=False)

(
    gg.ggplot(returns, gg.aes(x="Date", y="change"))
    + gg.geom_line(alpha=0.5, size=0.5)
    + gg.labs(x="date", y="daily returns", title="Volatility of the S&P 500 since 2008")
)
```

![png](001_getting-started-with-pymc3_files/001_getting-started-with-pymc3_8_0.png)

    <ggplot: (354016573)>

Use a `GaussianRandomWalk` as the prior for the latent volatilities.
It is a vector-valued distribution where the values of the vector form a random normal walk of length $n$, specified by the `shape` parameter.

We can provide initial values for any distribution, known as *test values*, using the `testval` parameter.
This can be useful if some values are illegal and we want to ensure a legal value is selected.

```python
with pm.Model() as sp500_model:
    change_returns = pm.Data(
        "returns", returns["change"], dims="date", export_index_as_coords=True
    )

    nu = pm.Exponential("nu", 1.0 / 10.0, testval=5.0)
    sigma = pm.Exponential("sigma", 2.0, testval=0.1)

    s = pm.GaussianRandomWalk("s", sigma=sigma, dims="date")

    volatility_process = pm.Deterministic(
        "volatility_process", pm.math.exp(-2 * s) ** 0.5, dims="date"
    )

    r = pm.StudentT(
        "r", nu=nu, sigma=volatility_process, observed=change_returns, dims="date"
    )
```

```python
pm.model_to_graphviz(sp500_model)
```

![svg](001_getting-started-with-pymc3_files/001_getting-started-with-pymc3_11_0.svg)

```python
sp500_model.RV_dims
```

    {'returns': ('date',),
     's': ('date',),
     'volatility_process': ('date',),
     'r': ('date',)}

```python
sp500_model.coords
```

    {'date': RangeIndex(start=0, stop=2906, step=1)}

```python
with sp500_model:
    trace = pm.sample(2000, init="adapt_diag", return_inferencedata=False)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [s, sigma, nu]

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
  <progress value='6000' class='' max='6000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [6000/6000 08:52<00:00 Sampling 2 chains, 0 divergences]
</div>

    Sampling 2 chains for 1_000 tune and 2_000 draw iterations (2_000 + 4_000 draws total) took 542 seconds.


    0, dim: date, 2906 =? 2906


    The rhat statistic is larger than 1.05 for some parameters. This indicates slight problems during sampling.
    The estimated number of effective samples is smaller than 200 for some parameters.

```python
az.plot_trace(trace, combined=False, var_names=["nu", "sigma"])
plt.show()
```

    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/arviz/data/io_pymc3.py:88: FutureWarning: Using `from_pymc3` without the model will be deprecated in a future release. Not using the model will return less accurate and less useful results. Make sure you use the model argument or call from_pymc3 within a model context.


    0, dim: date, 2906 =? 2906
    0, dim: date, 2906 =? 2906

![png](001_getting-started-with-pymc3_files/001_getting-started-with-pymc3_15_2.png)

```python
volatility_post = 1 / np.exp(trace["s", ::5].T)
volatility_post_hdi = az.hdi(volatility_post.T, hdi_prob=0.89)
plot_data = returns.copy()
plot_data["volatility"] = volatility_post.mean(axis=1)
plot_data["volatility_lower"] = volatility_post_hdi[:, 0]
plot_data["volatility_high"] = volatility_post_hdi[:, 1]

(
    gg.ggplot(plot_data, gg.aes(x="Date"))
    + gg.geom_line(gg.aes(y="change"), alpha=0.2, size=0.5)
    + gg.geom_ribbon(
        gg.aes(ymin="volatility_lower", ymax="volatility_high"),
        alpha=0.7,
        fill="#EF5BDE",
    )
    + gg.geom_line(gg.aes(y="volatility"), alpha=0.8, color="#A10D90", size=1)
    + gg.labs(x="date", y="daily returns", title="Volatility of the S&P 500 since 2008")
)
```

    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/arviz/stats/stats.py:493: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions

![png](001_getting-started-with-pymc3_files/001_getting-started-with-pymc3_16_1.png)

    <ggplot: (355534207)>

---

## Case Study 2: Coal mining disasters

Data is a time series of recorded coal mining disasters in the UK from 1851 to 1962.
The number of disasters is affected by changes in safety regulations.
There are two years with missing data, but *these missing values with be automatically imputed by PyMC3*.

Build a model to estimate when the change occurred and see how the model handles missing data.

```python
disasters_data = pd.read_csv("data/mining_disasters.txt", header=None)
disasters_data.columns = ["n_disasters"]
disasters_data[["year"]] = np.arange(1851, 1962)
disasters_data.head()
```

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
      <th>n_disasters</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.0</td>
      <td>1851</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>1852</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.0</td>
      <td>1853</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>1854</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1855</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Missing data
disasters_data[disasters_data.n_disasters.isna()]
```

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
      <th>n_disasters</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>39</th>
      <td>NaN</td>
      <td>1890</td>
    </tr>
    <tr>
      <th>83</th>
      <td>NaN</td>
      <td>1934</td>
    </tr>
  </tbody>
</table>
</div>

```python
(
    gg.ggplot(
        disasters_data[~disasters_data.n_disasters.isna()],
        gg.aes(x="year", y="n_disasters"),
    )
    + gg.geom_smooth(se=False, linetype="--", color="#BC272A")
    + gg.geom_line(group="a", alpha=0.2, color="#3F7BB1")
    + gg.geom_point(color="#3F7BB1")
    + gg.labs(x="year", y="disaster count")
)
```

![png](001_getting-started-with-pymc3_files/001_getting-started-with-pymc3_20_0.png)

    <ggplot: (356626845)>

The occurrence of disasters follows a Poisson process with a large rate parameter early on but a smaller parameter later.
We want to locate the change point in the series.

$$
\begin{align}
D_t & \sim \text{Pois}(r_t), r_t =
\begin{cases}
 e, & \text{if } t \leq s \\
 l, & \text{if } t \gt s
\end{cases} \\
s & \sim \text{Unif}(t_l, t_h) \\
e & \sim \exp(1) \\
l & \sim \exp(1)
\end{align}
$$

In this model, $s$ is the switchpoint between "early" and "late" rate parameters ($e$ and $l$).

```python

```

```python
with pm.Model() as disaster_model:
    # Switchpiont
    switchpoint = pm.DiscreteUniform(
        "switchpoint",
        lower=disasters_data.year.min(),
        upper=disasters_data.year.max(),
        testval=1900,
    )

    # Priors for early and late rates.
    early_rate = pm.Exponential("early_rate", 1.0)
    late_rate = pm.Exponential("late_rate", 1.0)

    # Allocate appropriate Poisson rates using the switchpoint.
    rate = pm.math.switch(switchpoint >= disasters_data.year, early_rate, late_rate)

    # Observed data.
    disasters = pm.Poisson("disasters", rate, observed=disasters_data.n_disasters)
```

    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/pymc3/model.py:1668: ImputationWarning: Data in disasters contains missing values and will be automatically imputed from the sampling distribution.

```python
pm.model_to_graphviz(disaster_model)
```

![svg](001_getting-started-with-pymc3_files/001_getting-started-with-pymc3_24_0.svg)

NUTS cannot be used to sample because the data is discrete.
Instead, need to use a Metropolis step method that implements adaptive Metropolis-Hastings.
PyMC3 handles this automatically.

```python
with disaster_model:
    trace = pm.sample(10000, random_seed=RANDOM_SEED)
```

    Multiprocess sampling (2 chains in 2 jobs)
    CompoundStep
    >CompoundStep
    >>Metropolis: [disasters_missing]
    >>Metropolis: [switchpoint]
    >NUTS: [late_rate, early_rate]

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
  <progress value='22000' class='' max='22000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [22000/22000 00:25<00:00 Sampling 2 chains, 0 divergences]
</div>

    Sampling 2 chains for 1_000 tune and 10_000 draw iterations (2_000 + 20_000 draws total) took 35 seconds.
    The number of effective samples is smaller than 10% for some parameters.

```python
disaster_az = az.from_pymc3(trace=trace, model=disaster_model)
```

```python
az.plot_trace(disaster_az)
plt.show()
```

![png](001_getting-started-with-pymc3_files/001_getting-started-with-pymc3_28_0.png)

Posterior distribution for the switchpoint.

```python
switchpoint_post = pd.DataFrame({"switchpoint": trace["switchpoint"]})
(
    gg.ggplot(switchpoint_post, gg.aes(x="switchpoint"))
    + gg.geom_histogram(binwidth=1, alpha=0.4, size=0.5, color="black", fill="black")
    + gg.scale_x_continuous(breaks=disasters_data.year)
    + gg.scale_y_continuous(expand=(0, 0, 0.02, 0))
    + gg.theme(axis_text_x=gg.element_text(angle=45, ha="right"))
)
```

![png](001_getting-started-with-pymc3_files/001_getting-started-with-pymc3_30_0.png)

    <ggplot: (356303463)>

Posterior predictions automatically generated for missing data.

```python
missing_data_imputation = pd.DataFrame(
    trace["disasters_missing"], columns=["missing data 1", "missing data 2"]
).melt()

(
    gg.ggplot(missing_data_imputation, gg.aes(x="value"))
    + gg.facet_wrap("variable", nrow=1)
    + gg.geom_histogram(binwidth=1, alpha=0.4, size=0.5, color="black", fill="black")
    + gg.scale_y_continuous(expand=(0, 0, 0.02, 0))
    + gg.labs(
        x="expected number of disasters",
        y="count",
        title="Posterior predictions for missing data",
    )
)
```

    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/plotnine/facets/facet_wrap.py:215: UserWarning: This figure was using constrained_layout==True, but that is incompatible with subplots_adjust and or tight_layout: setting constrained_layout==False.

![png](001_getting-started-with-pymc3_files/001_getting-started-with-pymc3_32_1.png)

    <ggplot: (356505432)>

The following plot shows the results of the fit model on top of the original data.
The switchpoint is shown in dark red with its 89% CI in dashed lines and the black dashed line is the accident rate.
The estimates for the imputed data are also shown as green points with 89% CI.

```python
# Switchpoint summary stats.
switchpoint_mean = trace["switchpoint"].mean()
switchpoint_hdi = az.hdi(trace["switchpoint"], hdi_prob=0.89)

# Accident rate over all years.
average_disasters = np.zeros_like(disasters_data.n_disasters, dtype="float")
for i, year in enumerate(disasters_data.year):
    idx = year < trace["switchpoint"]
    average_disasters[i] = np.mean(
        np.where(idx, trace["early_rate"], trace["late_rate"])
    )

average_disasters_df = pd.DataFrame(
    {"avg_disasters": average_disasters, "year": disasters_data.year}
)

# Imputed data summary.
disaster_impute_summary = az.summary(
    trace, var_names="disasters_missing", hdi_prob=0.89, kind="stats"
)
disaster_impute_summary["year"] = disasters_data[
    disasters_data.n_disasters.isna()
].year.values

# Plotting
switchpoint_color = "#BC272A"
impute_color = "#12853B"
(
    gg.ggplot(disasters_data, gg.aes(x="year"))
    + gg.geom_point(gg.aes(y="n_disasters"), color="#3F7BB1", size=1, alpha=0.5)
    + gg.geom_line(
        gg.aes(y="avg_disasters"), data=average_disasters_df, linetype="--", size=1
    )
    + gg.geom_vline(xintercept=switchpoint_hdi, linetype="--", color=switchpoint_color)
    + gg.geom_vline(xintercept=switchpoint_mean, color=switchpoint_color, size=1)
    + gg.geom_linerange(
        gg.aes(ymin="hdi_5.5%", ymax="hdi_94.5%"),
        data=disaster_impute_summary,
        color=impute_color,
        size=0.8,
        alpha=0.5,
    )
    + gg.geom_point(
        gg.aes(y="mean"),
        data=disaster_impute_summary,
        color=impute_color,
        size=2.5,
        alpha=0.8,
    )
    + gg.scale_y_continuous(expand=(0.01, 0, 0.02, 0))
    + gg.labs(x="year", y="number of accidents")
)
```

    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/arviz/data/io_pymc3.py:88: FutureWarning: Using `from_pymc3` without the model will be deprecated in a future release. Not using the model will return less accurate and less useful results. Make sure you use the model argument or call from_pymc3 within a model context.
    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/plotnine/layer.py:467: PlotnineWarning: geom_point : Removed 2 rows containing missing values.

![png](001_getting-started-with-pymc3_files/001_getting-started-with-pymc3_34_1.png)

    <ggplot: (357116335)>

---

## Arbitrary deterministics

PyMC3 provides many mathematical functions and operators for transofmring random variables into new random variables.
Since it is not exhaustive, Theano and PyMC3 provide functionality for creating arbitary Theano functinos in Python that can be used in PyMC3 models.
This is supported throught the `@as_op` function operator.

The main limitation, though, is that Theano cannot inspect these functions to compute the gradient required for Hamiltonian-based samplers.
This means it is not possible to use the HMC nor NUTS samplers for models that use the function.
A work around is to inherit from `theno.Op` instead of using the `@as_op` decorator.

```python
import theano.tensor as tt
from theano.compile.ops import as_op


@as_op(itypes=[tt.lscalar], otypes=[tt.lscalar])
def crazy_modulo3(value):
    if value > 0:
        return value % 3
    else:
        return (-value + 1) % 3


with pm.Model() as model_deterministic:
    a = pm.Poisson("a", 1)
    b = crazy_modulo3(a)
```

---

## Generalized Linear Models

PyMC3 provides the `glm` submodule for constructing GLMs using the R formula language (using the [patsy](https://patsy.readthedocs.io/en/latest/) library).

```python
from pymc3.glm import GLM

d = pd.DataFrame({"X1": X1, "X2": X2, "Y": Y})

with pm.Model() as model_glm:
    GLM.from_formula("Y ~ X1 + X2", d)
    trace = pm.sample()
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [sd, X2, X1, Intercept]

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
  100.00% [4000/4000 00:06<00:00 Sampling 2 chains, 0 divergences]
</div>

    Sampling 2 chains for 1_000 tune and 1_000 draw iterations (2_000 + 2_000 draws total) took 15 seconds.

Below is an example of a logisitic model.

```python
from pymc3.glm.families import Binomial

d_logistic = d.assign(Y=lambda d: d.Y > np.median(d.Y))

with pm.Model() as model_glm_logistic:
    GLM.from_formula("Y ~ X1", d_logistic, family=Binomial())
    trace = pm.sample()
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [X1, Intercept]

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
  100.00% [4000/4000 00:06<00:00 Sampling 2 chains, 0 divergences]
</div>

    Sampling 2 chains for 1_000 tune and 1_000 draw iterations (2_000 + 2_000 draws total) took 14 seconds.
    The acceptance probability does not match the target. It is 0.8891785452120076, but should be close to 0.8. Try to increase the number of tuning steps.

The [Bambi](https://bambinos.github.io/bambi/) library provides a similar interface for designing hierarchical models.

---

```python
%load_ext watermark
%watermark -n -u -v -iv -w
```

    Last updated: Tue Feb 02 2021

    Python implementation: CPython
    Python version       : 3.9.1
    IPython version      : 7.20.0

    theano    : 1.0.5
    numpy     : 1.20.0
    arviz     : 0.11.0
    matplotlib: 3.3.4
    plotnine  : 0.7.1
    pandas    : 1.2.1
    pymc3     : 3.9.3

    Watermark: 2.1.0

```python

```
