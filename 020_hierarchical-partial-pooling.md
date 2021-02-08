# Hierarchical Partial Pooling

**[Original tutorial](https://docs.pymc.io/notebooks/hierarchical_partial_pooling.html)**

Estimate batting average of several players, but they each have a different number of at-bats.

## Approach

We will use PyMC3 to estimate the batting average for 18 baseball players ([Efron and Morris](http://www.swarthmore.edu/NatSci/peverso1/Sports%20Data/JamesSteinData/Efron-Morris%20Baseball/EfronMorrisBB.txt).
Having estimated the averages across all players, we will then estimate the batting average of a new player with 4 at-bats and no hits.

We will use a hierarhcical structure to share batting average information across players.

```python
import arviz as az
import janitor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import pymc3 as pm
import theano.tensor as tt

gg.theme_set(gg.theme_minimal)
%config InlineBackend.figure_format = "retina"
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
```

## Data

```python
data = pd.read_csv(pm.get_data("efron-morris-75-data.tsv"), sep="\t").clean_names()
data
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
      <th>firstname</th>
      <th>lastname</th>
      <th>at_bats</th>
      <th>hits</th>
      <th>battingaverage</th>
      <th>remainingat_bats</th>
      <th>remainingaverage</th>
      <th>seasonat_bats</th>
      <th>seasonhits</th>
      <th>seasonaverage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Roberto</td>
      <td>Clemente</td>
      <td>45</td>
      <td>18</td>
      <td>0.400</td>
      <td>367</td>
      <td>0.3460</td>
      <td>412</td>
      <td>145</td>
      <td>0.352</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Frank</td>
      <td>Robinson</td>
      <td>45</td>
      <td>17</td>
      <td>0.378</td>
      <td>426</td>
      <td>0.2981</td>
      <td>471</td>
      <td>144</td>
      <td>0.306</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Frank</td>
      <td>Howard</td>
      <td>45</td>
      <td>16</td>
      <td>0.356</td>
      <td>521</td>
      <td>0.2764</td>
      <td>566</td>
      <td>160</td>
      <td>0.283</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jay</td>
      <td>Johnstone</td>
      <td>45</td>
      <td>15</td>
      <td>0.333</td>
      <td>275</td>
      <td>0.2218</td>
      <td>320</td>
      <td>76</td>
      <td>0.238</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ken</td>
      <td>Berry</td>
      <td>45</td>
      <td>14</td>
      <td>0.311</td>
      <td>418</td>
      <td>0.2727</td>
      <td>463</td>
      <td>128</td>
      <td>0.276</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Jim</td>
      <td>Spencer</td>
      <td>45</td>
      <td>14</td>
      <td>0.311</td>
      <td>466</td>
      <td>0.2704</td>
      <td>511</td>
      <td>140</td>
      <td>0.274</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Don</td>
      <td>Kessinger</td>
      <td>45</td>
      <td>13</td>
      <td>0.289</td>
      <td>586</td>
      <td>0.2645</td>
      <td>631</td>
      <td>168</td>
      <td>0.266</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Luis</td>
      <td>Alvarado</td>
      <td>45</td>
      <td>12</td>
      <td>0.267</td>
      <td>138</td>
      <td>0.2101</td>
      <td>183</td>
      <td>41</td>
      <td>0.224</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ron</td>
      <td>Santo</td>
      <td>45</td>
      <td>11</td>
      <td>0.244</td>
      <td>510</td>
      <td>0.2686</td>
      <td>555</td>
      <td>148</td>
      <td>0.267</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Ron</td>
      <td>Swaboda</td>
      <td>45</td>
      <td>11</td>
      <td>0.244</td>
      <td>200</td>
      <td>0.2300</td>
      <td>245</td>
      <td>57</td>
      <td>0.233</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Rico</td>
      <td>Petrocelli</td>
      <td>45</td>
      <td>10</td>
      <td>0.222</td>
      <td>538</td>
      <td>0.2639</td>
      <td>583</td>
      <td>152</td>
      <td>0.261</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Ellie</td>
      <td>Rodriguez</td>
      <td>45</td>
      <td>10</td>
      <td>0.222</td>
      <td>186</td>
      <td>0.2258</td>
      <td>231</td>
      <td>52</td>
      <td>0.225</td>
    </tr>
    <tr>
      <th>12</th>
      <td>George</td>
      <td>Scott</td>
      <td>45</td>
      <td>10</td>
      <td>0.222</td>
      <td>435</td>
      <td>0.3034</td>
      <td>480</td>
      <td>142</td>
      <td>0.296</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Del</td>
      <td>Unser</td>
      <td>45</td>
      <td>10</td>
      <td>0.222</td>
      <td>277</td>
      <td>0.2635</td>
      <td>322</td>
      <td>83</td>
      <td>0.258</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Billy</td>
      <td>Williams</td>
      <td>45</td>
      <td>10</td>
      <td>0.222</td>
      <td>591</td>
      <td>0.3299</td>
      <td>636</td>
      <td>205</td>
      <td>0.251</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Bert</td>
      <td>Campaneris</td>
      <td>45</td>
      <td>9</td>
      <td>0.200</td>
      <td>558</td>
      <td>0.2849</td>
      <td>603</td>
      <td>168</td>
      <td>0.279</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Thurman</td>
      <td>Munson</td>
      <td>45</td>
      <td>8</td>
      <td>0.178</td>
      <td>408</td>
      <td>0.3162</td>
      <td>453</td>
      <td>137</td>
      <td>0.302</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Max</td>
      <td>Alvis</td>
      <td>45</td>
      <td>7</td>
      <td>0.156</td>
      <td>70</td>
      <td>0.2000</td>
      <td>115</td>
      <td>21</td>
      <td>0.183</td>
    </tr>
  </tbody>
</table>
</div>

## Model

We will assume there exists a hidden factor $\phi \in [0, 1]$ related to the expected performance for all players.
We know nothing about this value and will thus us an uninformative prior: $\phi \sim \text{Uniform}(0, 1)$.

Another hyperparameter $\kappa$ will account for the variance in the population of batting averages.
We will use a bounded Pareto distribution as a prior to ensure the estimated value falls within a reasonable range.
However, the Pareto distribution is difficult to sample, so will use a simple trick based on the fact that the log of a Pareto distributed random variable follows an exponential distirbution.

Both of these hyperparameters will be used to parameterize a Beta distribution because it is ideal for modeling quantities on the unit interval $[0, 1]$.
Usually, the Beta distribution is parameterized via a scale and shape parameter, but it can also be parameterized in terms of its mean $\mu \in [0, 1]$ and sample size (as a proxy for variance): $\nu = \alpha + \beta(\nu > 0)$.

Finally, the sampling distribution will be a Binomial where each player either hit or missed at each at-bat.

```python
at_bats, hits = data[["at_bats", "hits"]].to_numpy().T
N = len(hits)

with pm.Model() as baseball_model:

    at_bats_shared = pm.Data("at_bats", data.at_bats.values)
    hits_shared = pm.Data("hits", data.hits.values)

    ϕ = pm.Uniform("ϕ", lower=0.0, upper=1.0)

    κ_log = pm.Exponential("κ_log", lam=1.5)
    κ = pm.Deterministic("κ", tt.exp(κ_log))

    θ = pm.Beta("θ", alpha=ϕ * κ, beta=(1.0 - ϕ) * κ, shape=N)

    y = pm.Binomial("y", n=at_bats_shared, p=θ, observed=hits_shared)

    θ_new = pm.Beta("θ_new", alpha=ϕ * κ, beta=(1.0 - ϕ) * κ)
    y_new = pm.Binomial("y_new", n=4, p=θ_new, observed=0)

    trace = pm.sample(
        2000,
        init="advi",
        n_init=10000,
        tune=2500,
        chains=2,
        cores=2,
        target_accept=0.95,
        random_seed=RANDOM_SEED,
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
  <progress value='6906' class='' max='10000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  69.06% [6906/10000 00:02<00:01 Average Loss = 78.75]
</div>

    Convergence achieved at 7100
    Interrupted at 7,099 [70%]: Average Loss = 115.6
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [θ_new, θ, κ_log, ϕ]

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
  <progress value='8000' class='' max='8000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [8000/8000 00:50<00:00 Sampling 2 chains, 10 divergences]
</div>

    Sampling 2 chains for 2_000 tune and 2_000 draw iterations (4_000 + 4_000 draws total) took 61 seconds.
    There were 10 divergences after tuning. Increase `target_accept` or reparameterize.
    The number of effective samples is smaller than 10% for some parameters.

```python
pm.model_to_graphviz(baseball_model)
```

![svg](020_hierarchical-partial-pooling_files/020_hierarchical-partial-pooling_6_0.svg)

```python
trace_az = az.from_pymc3(trace=trace, model=baseball_model)
```

```python
theta_post = az.summary(trace_az, var_names="θ", hdi_prob=0.89)
theta_post["player"] = [f + " " + l for f, l in zip(data.firstname, data.lastname)]
theta_post["player"] = pd.Categorical(
    theta_post.player.values, categories=theta_post.player.values, ordered=True
)

(
    gg.ggplot(theta_post, gg.aes(x="player", y="mean"))
    + gg.geom_linerange(gg.aes(ymin="hdi_5.5%", ymax="hdi_94.5%"))
    + gg.geom_point()
    + gg.coord_flip()
    + gg.theme(axis_title_y=gg.element_blank())
)
```

![png](020_hierarchical-partial-pooling_files/020_hierarchical-partial-pooling_8_0.png)

    <ggplot: (365291653)>

```python
az.plot_posterior(trace_az, var_names=["ϕ", "κ"], hdi_prob=0.89)
plt.show()
```

![png](020_hierarchical-partial-pooling_files/020_hierarchical-partial-pooling_9_0.png)

```python
with baseball_model:
    ppc = pm.sample_posterior_predictive(trace=trace, samples=100)
```

    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/pymc3/sampling.py:1707: UserWarning: samples parameter is smaller than nchains times ndraws, some draws and/or chains may not be represented in the returned posterior predictive sample

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
  <progress value='100' class='' max='100' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [100/100 00:01<00:00]
</div>

```python
ppc_df = pd.DataFrame(ppc["y"], columns=theta_post.player).melt()
ppc_df["player"] = pd.Categorical(ppc_df.player, categories=theta_post.player)

(
    gg.ggplot(ppc_df, gg.aes(x="player", y="value"))
    + gg.geom_boxplot(fill="grey", alpha=0.2)
    + gg.coord_flip()
    + gg.theme(axis_title_y=gg.element_blank())
    + gg.labs(y="predicted number of hits")
)
```

![png](020_hierarchical-partial-pooling_files/020_hierarchical-partial-pooling_11_0.png)

    <ggplot: (363772260)>

```python
ax = az.plot_posterior(trace_az, var_names="θ_new", hdi_prob=0.89)
ax.set_title("Predicted θ")
plt.show()
```

![png](020_hierarchical-partial-pooling_files/020_hierarchical-partial-pooling_12_0.png)

```python
(
    gg.ggplot(pd.DataFrame({"pred": ppc["y_new"]}), gg.aes(x="pred"))
    + gg.geom_bar()
    + gg.scale_y_continuous(expand=(0, 0, 0.02, 0))
    + gg.labs(x="predicted number of hits")
)
```

![png](020_hierarchical-partial-pooling_files/020_hierarchical-partial-pooling_13_0.png)

    <ggplot: (363210882)>

---

```python
%load_ext watermark
%watermark -n -u -v -iv -w
```

    Last updated: Mon Feb 08 2021

    Python implementation: CPython
    Python version       : 3.9.1
    IPython version      : 7.20.0

    arviz     : 0.11.0
    theano    : 1.0.5
    numpy     : 1.20.0
    pandas    : 1.2.1
    plotnine  : 0.7.1
    pymc3     : 3.9.3
    janitor   : 0.20.10
    matplotlib: 3.3.4

    Watermark: 2.1.0

```python

```
