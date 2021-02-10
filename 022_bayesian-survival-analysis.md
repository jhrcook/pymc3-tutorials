# Bayesian Survival Analysis

**[Original tutorial](https://docs.pymc.io/notebooks/survival_analysis.html)**

Survival analysis studies the distribution of the time to an event.
This tutorial will use data on mastectomy results.

```python
from typing import Callable

import arviz as az
import numpy as np
import pandas as pd
import plotnine as gg
import pymc3 as pm
import scipy.stats as st
import seaborn as sns
from matplotlib import pyplot as plt
from pymc3.distributions.timeseries import GaussianRandomWalk
from theano import tensor as tt

gg.theme_set(gg.theme_minimal)
%config InlineBackend.figure_format = "retina"
RANDOM_SEED = 601
np.random.seed(RANDOM_SEED)
```

## Data

Each row in the data represents an observation from a woman diagnosed with breast cancer that underwent a mastectomy.
The column `time` represents the months post-surgery that the woman was observed.
The column `event` indicates whether or not the woman died during the observation period.
The column `metastized` indicates whether the cancer had metastized prior to surgery.

```python
df = pd.read_csv(pm.get_data("mastectomy.csv"))
df["event"] = df.event.astype(np.int64)
df["metastized"] = (df.metastized == "yes").astype(np.int64)
n_patients = df.shape[0]
patients = np.arange(n_patients)

df.head()
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
      <th>time</th>
      <th>event</th>
      <th>metastized</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>47</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>69</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>70</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

```python
n_patients
```

    44

## A "crash course" on survival analysis

If the random variable $T$ is the time to the event we are studying, then survival anylsis is concerned with the survival function

$$
S(t) = P(T > t) = 1 - F(t)
$$

where $F$ is the CDF of $T$.
It is convenient the express the survival function in terms of the **hazard rate** $\lambda(t)$: the instantaneous propbability that the event occurs at time $t$ given it has yet to occur.
That is,

$$
\begin{align}
\lambda(t) &= \lim_{\Delta t \to 0} \frac{P(t < T < t + \Delta t | T > t)}{\Delta t} \\
&= \lim_{\Delta t \to 0} \frac{P(t < T < t + \Delta t)}{\Delta t \cdot P(T > t)} \\
&= \frac{1}{S(t)} \cdot \lim_{\Delta t \to 0} \frac{S(t + \Delta t) - S(t)}{\Delta t} = - \frac{S'(t)}{S(t)} \\
\end{align}
$$

Solving this differential equation for the survival function demonstrates

$$
S(t) = \exp \lgroup - \int_{0}^{s} \lambda(s) ds \rgroup
$$

Which in turn shows that the cumulative hazard function

$$
\Lambda(t) = \int_0^t \lambda(s) ds
$$

is an important quantity in survival analysis which we can concisely write as

$$
S(t) = \exp(- \Lambda(t))
$$

An important part of survival analysis is **censoring** because we will not observe the death of every subject.
For our data, the column `event` is 1 if the subject's death ws observed and a 0 is the death was not observed (censored).
The time for a censored data point is *not* the subject's survival time, but, instead, we assume the subject's true survival time exceeds the time point.

```python
df.head()
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
      <th>time</th>
      <th>event</th>
      <th>metastized</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>47</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>69</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>70</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

```python
plot_data = df.copy().reset_index(drop=False).rename(columns={"index": "subject_id"})

(
    gg.ggplot(plot_data, gg.aes(x="subject_id", y="time"))
    + gg.geom_linerange(gg.aes(ymax="time", color="factor(1-event)"), ymin=0)
    + gg.geom_point(gg.aes(alpha="factor(metastized)"))
    + gg.scale_color_brewer(
        type="qual", palette="Set1", labels=("uncensored", "censored")
    )
    + gg.scale_alpha_manual(values=(0, 1), labels=("not metastized", "metastized"))
    + gg.coord_flip()
    + gg.theme(legend_title=gg.element_blank())
    + gg.labs(x="subject", y="months since mastectomy")
)
```

![png](022_bayesian-survival-analysis_files/022_bayesian-survival-analysis_7_0.png)

    <ggplot: (280873674)>

## Bayesian proportional hazards model

Since our goal is to understand the impact of metastization on survival time, a risk regression model is appropriate.
The most commonly used risk regression model is the Cox's proportional hazards model.
In this model, with covariates $\textbf{x}$ and regression coefficients $\beta$, the hazard rate is modeled as

$$
\lambda(t) = \lambda_0 (t) \exp(\textbf{x} \beta)
$$

Here, $\lambda_0 (t)$ is the baseline hazard which is independent of the covariates $\textbf{x}$.

We need to specify priors for $\beta$ and $\lambda_0 (t)$.
For $\beta$, we can use a normal distribution $N(\mu_\beta, \sigma^2_\beta)$ and place uniform priors on those parameters.

Since $\lambda_0 (t)$ is a piecewise constant function, its prior will be a semiparametric prior that requires us to partition the time range into intervals with endpoints $0 \leq s_1 \lt s_2 \lt \dots \lt s_N$.
With this partition, $\lambda_0 (t) = \lambda_j$ if $s_j \leq t \lt s_{j+1}$.
With $\lambda_0 (t)$ constrained to have this form, we just need to provide priors for the N-1 values $\lambda_j$.
We can use independent vague priors $\lambda_j \sim \text{Gamma}(10^{-2}, 10^{-2})$ and each interval will be 3 months long.

```python
interval_length = 3
interval_bounds = np.arange(0, df.time.max() + interval_length + 1, interval_length)
n_intervals = interval_bounds.size - 1
intervals = np.arange(n_intervals)
```

```python
(
    gg.ggplot(df, gg.aes(x="time"))
    + gg.geom_histogram(gg.aes(fill="factor(1- event)"), breaks=interval_bounds)
    + gg.scale_fill_brewer(
        type="qual", palette="Set1", labels=("uncensored", "censored")
    )
    + gg.scale_y_continuous(expand=(0, 0, 0.02, 0))
    + gg.theme(legend_title=gg.element_blank())
    + gg.labs(x="months since mastectomy", y="number of observations")
)
```

![png](022_bayesian-survival-analysis_files/022_bayesian-survival-analysis_10_0.png)

    <ggplot: (352422051)>

```python
x = np.linspace(0, 20, 200)
y = st.gamma.pdf(x, 1, scale=1.0 / 1)
gamma_plot_data = pd.DataFrame({"x": x, "y": y})
gamma_plot_data = gamma_plot_data[gamma_plot_data.y > 0.001]
(
    gg.ggplot(gamma_plot_data, gg.aes("x", "y"))
    + gg.geom_line(group="a", size=1.2)
    + gg.scale_x_continuous(expand=(0, 0, 0, 0), limits=(0, np.nan))
    + gg.scale_y_continuous(expand=(0, 0, 0, 0), limits=(0, np.nan))
    + gg.labs(x="x", y="probability density", title="Gamma distribution")
)
```

![png](022_bayesian-survival-analysis_files/022_bayesian-survival-analysis_11_0.png)

    <ggplot: (364509049)>

We can now figure out how to fit the model using MCMC simulation.
The piecewise-constant proportional hazard model is closely related to a Poisson regression model.
"The models are not identical, but their likelihoods differ by a factor that depends only on the observed data and not the parameters $\beta$ and $\lambda_j$.
For details, see Germán Rodríguez’s WWS 509 [course notes](http://data.princeton.edu/wws509/notes/c7s4.html)."

We will use an indicator variable based on whether or not the $i$-th subject died in the $j$-th interval

$$
d_{i,j} =
\begin{cases}
  1 &\text{if subject } i \text{ died in interval } j \\
  0 &\text{otherwise}
\end{cases}
$$

```python
last_period = np.floor((df.time - 0.01) / interval_length).astype(int)
death = np.zeros((n_patients, n_intervals))
death[patients, last_period] = df.event
```

Also, define $t_{i,j}$ to be the amount of time the $i$-th subject was at risk in the $j$-th interval.

```python
exposure = (
    np.greater_equal.outer(df.time.to_numpy(), interval_bounds[:-1]) * interval_length
)
exposure[patients, last_period] = df.time - interval_bounds[last_period]
```

```python
with pm.Model() as model:
    λ0 = pm.Gamma("λ0", 1.0, 1.0, shape=n_intervals)

    β = pm.Normal("β", 0, sigma=5)

    λ = pm.Deterministic("λ", tt.outer(tt.exp(β * df.metastized), λ0))
    μ = pm.Deterministic("μ", exposure * λ)

    obs = pm.Poisson("obs", μ, observed=death)
```

```python
pm.model_to_graphviz(model)
```

![svg](022_bayesian-survival-analysis_files/022_bayesian-survival-analysis_17_0.svg)

```python
with model:
    trace = pm.sample(
        1000, init="advi", n_init=50000, tune=1000, random_seed=RANDOM_SEED
    )
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using advi...

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
  <progress value='20205' class='' max='50000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  40.41% [20205/50000 00:05<00:08 Average Loss = 366.98]
</div>

    Convergence achieved at 20400
    Interrupted at 20,399 [40%]: Average Loss = 937.76
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [β, λ0]

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
  100.00% [4000/4000 00:14<00:00 Sampling 2 chains, 0 divergences]
</div>

    Sampling 2 chains for 1_000 tune and 1_000 draw iterations (2_000 + 2_000 draws total) took 24 seconds.

```python
trace_az = az.from_pymc3(trace=trace, model=model)
```

```python
az.plot_trace(trace_az, var_names=["λ0", "β"]);
```

![png](022_bayesian-survival-analysis_files/022_bayesian-survival-analysis_20_0.png)

```python
az.plot_posterior(trace_az, var_names="β");
```

![png](022_bayesian-survival-analysis_files/022_bayesian-survival-analysis_21_0.png)

```python
az.plot_autocorr(trace_az, var_names="β");
```

![png](022_bayesian-survival-analysis_files/022_bayesian-survival-analysis_22_0.png)

## Analysis of coefficients

We will now examine the effect of metatstization on the cumulative hazard function and survival function.

```python
base_hazard = trace["λ0"]
met_hazard = trace["λ0"] * np.exp(np.atleast_2d(trace["β"]).T)
```

```python
def cum_hazard(hazard: np.ndarray) -> np.ndarray:
    return (interval_length * hazard).cumsum(axis=-1)


def survival(hazard: np.ndarray) -> np.ndarray:
    return np.exp(-cum_hazard(hazard))


def build_summary_dataframe(
    x: np.ndarray,
    hazard: np.ndarray,
    f: Callable,
    hdi_prob: float = 0.89,
) -> pd.DataFrame:
    hazard_mean = f(hazard.mean(axis=0))
    hazard_hpd = az.hpd(f(hazard), hdi_prob=hdi_prob)

    plot_data = pd.DataFrame(
        {
            "x": x,
            "hazard_mean": hazard_mean,
            "hpd_low": hazard_hpd[:, 0],
            "hpd_high": hazard_hpd[:, 1],
        }
    )

    return plot_data
```

```python
hazard_plot_data = pd.concat(
    [
        build_summary_dataframe(interval_bounds[:-1], base_hazard, f=cum_hazard).assign(
            metastized=False
        ),
        build_summary_dataframe(interval_bounds[:-1], met_hazard, f=cum_hazard).assign(
            metastized=True
        ),
    ]
)

hazard_plot_data.head()
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
      <th>x</th>
      <th>hazard_mean</th>
      <th>hpd_low</th>
      <th>hpd_high</th>
      <th>metastized</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.054854</td>
      <td>0.000010</td>
      <td>0.122975</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0.167472</td>
      <td>0.020913</td>
      <td>0.291957</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>0.280649</td>
      <td>0.079634</td>
      <td>0.456071</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>0.395625</td>
      <td>0.150241</td>
      <td>0.600450</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>0.512123</td>
      <td>0.237538</td>
      <td>0.762393</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>

```python
pal_kwargs = {
    "type": "qual",
    "palette": "Set1",
    "labels": ("not metastized", "metastized"),
}
(
    gg.ggplot(hazard_plot_data, gg.aes(x="x"))
    + gg.geom_ribbon(
        gg.aes(ymin="hpd_low", ymax="hpd_high", fill="metastized"), alpha=0.3
    )
    + gg.geom_line(gg.aes(y="hazard_mean", color="metastized"))
    + gg.scale_color_brewer(**pal_kwargs)
    + gg.scale_fill_brewer(**pal_kwargs)
    + gg.theme(legend_title=gg.element_blank())
    + gg.labs(x="months since mastectomy", y=r"cumulative hazard $\Lambda(t)$")
),
```

![png](022_bayesian-survival-analysis_files/022_bayesian-survival-analysis_27_0.png)

    (<ggplot: (356960826)>,)

```python

```
