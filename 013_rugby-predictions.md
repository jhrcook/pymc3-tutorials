# A Hierarchical model for rugby prediction

Our goal is to infer a latent parameter for the "strength" of a team based on their scoring intensity.

```python
from io import StringIO

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import pymc3 as pm
import seaborn as sns
import theano.tensor as tt

%matplotlib inline
%config InlineBackend.figure_format = "retina"

gg.theme_set(gg.theme_minimal())

RANDOM_SEED = 847
np.random.seed(RANDOM_SEED)
```

## Data

```python
df_all = pd.read_csv(pm.get_data("rugby.csv"), index_col=0)

df_all["difference"] = np.abs(df_all["home_score"] - df_all["away_score"])
df_all["difference_non_abs"] = df_all["home_score"] - df_all["away_score"]

df_all.head()
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
      <th>home_team</th>
      <th>away_team</th>
      <th>home_score</th>
      <th>away_score</th>
      <th>year</th>
      <th>difference</th>
      <th>difference_non_abs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wales</td>
      <td>Italy</td>
      <td>23</td>
      <td>15</td>
      <td>2014</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>France</td>
      <td>England</td>
      <td>26</td>
      <td>24</td>
      <td>2014</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ireland</td>
      <td>Scotland</td>
      <td>28</td>
      <td>6</td>
      <td>2014</td>
      <td>22</td>
      <td>22</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ireland</td>
      <td>Wales</td>
      <td>26</td>
      <td>3</td>
      <td>2014</td>
      <td>23</td>
      <td>23</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Scotland</td>
      <td>England</td>
      <td>0</td>
      <td>20</td>
      <td>2014</td>
      <td>20</td>
      <td>-20</td>
    </tr>
  </tbody>
</table>
</div>

## Visualization / EDA

```python
(
    gg.ggplot(df_all, gg.aes(x="factor(year)", y="difference"))
    + gg.geom_boxplot(fill="black", alpha=0.2, outlier_alpha=0)
    + gg.labs(
        x="year",
        y="Average (abs) point difference",
        title="Average magnitude of scores differences of Six Nations",
    )
)
```

![png](013_rugby-predictions_files/013_rugby-predictions_5_0.png)

    <ggplot: (273029494)>

```python
df_all.head()
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
      <th>home_team</th>
      <th>away_team</th>
      <th>home_score</th>
      <th>away_score</th>
      <th>year</th>
      <th>difference</th>
      <th>difference_non_abs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wales</td>
      <td>Italy</td>
      <td>23</td>
      <td>15</td>
      <td>2014</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>France</td>
      <td>England</td>
      <td>26</td>
      <td>24</td>
      <td>2014</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ireland</td>
      <td>Scotland</td>
      <td>28</td>
      <td>6</td>
      <td>2014</td>
      <td>22</td>
      <td>22</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ireland</td>
      <td>Wales</td>
      <td>26</td>
      <td>3</td>
      <td>2014</td>
      <td>23</td>
      <td>23</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Scotland</td>
      <td>England</td>
      <td>0</td>
      <td>20</td>
      <td>2014</td>
      <td>20</td>
      <td>-20</td>
    </tr>
  </tbody>
</table>
</div>

We can see that Italy and England have poor away-form.

```python
plot_data = (
    df_all[["difference_non_abs", "home_team", "away_team", "year"]]
    .melt(id_vars=["difference_non_abs", "year"])
    .assign(variable=lambda d: [s.replace("_team", "") for s in d.variable])
)

(
    gg.ggplot(plot_data, gg.aes(x="value", y="difference_non_abs"))
    + gg.geom_hline(yintercept=0, linetype="--", alpha=0.5, color="black")
    + gg.geom_boxplot(
        gg.aes(color="variable", fill="variable"),
        position="dodge",
        alpha=0.2,
        outlier_alpha=0.3,
    )
    + gg.scale_color_brewer(type="qual", palette="Set1")
    + gg.scale_fill_brewer(type="qual", palette="Set1")
    + gg.theme(legend_title=gg.element_blank())
    + gg.labs(x=None, y="Score difference (home score - away score)")
)
```

![png](013_rugby-predictions_files/013_rugby-predictions_8_0.png)

    <ggplot: (342032255)>

```python

```

```python
plot_data_summary = (
    plot_data.groupby(["year", "variable", "value"]).mean().reset_index(drop=False)
)

(
    gg.ggplot(plot_data, gg.aes(x="year", y="difference_non_abs"))
    + gg.facet_grid("value ~ variable")
    + gg.geom_hline(yintercept=0, linetype="--", color="black", alpha=0.5)
    + gg.geom_line(
        group="a", data=plot_data_summary, alpha=0.7, color="#3F7BB1", size=1
    )
    + gg.geom_point()
    + gg.theme(figure_size=(8, 12))
)
```

    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/plotnine/facets/facet_grid.py:136: FutureWarning: Index.__and__ operating as a set operation is deprecated, in the future this will be a logical operation matching Series.__and__.  Use index.intersection(other) instead
    /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/plotnine/facets/facet_grid.py:137: FutureWarning: Index.__and__ operating as a set operation is deprecated, in the future this will be a logical operation matching Series.__and__.  Use index.intersection(other) instead

![png](013_rugby-predictions_files/013_rugby-predictions_10_1.png)

    <ggplot: (342163505)>

## Model

There are 6 teams ($T=6$) that play each other once in a season.
The number of points scored by the home and away team in the $g$-th game of the season (15 games in total) as $y_{g1}$ and $y_{g2}$.
The vector of observed counts $y = (y_{g1}, y_{g2})$ is modelled as an independent Poisson: $\mathbb{y}_{gi} | \theta_{gj} \sim \text{Poisson}(\theta_{gj})$ where the $\theta$ parameters represents the scoring intensity in the $g$-th game for the team playing at home ($j=1$) or away ($j=2$).

The model is shown below constructed of two equations, one for the number of points by the home team and one for the away team.
The only differences is that the home team has the $home$ covariate to include for the "home-field advantage".

$$
\begin{align}
\log \theta_{g1} &= home + att_{h(g)} + def_{a(g)} \\
\log \theta_{g2} &= att_{a(g)} + def_{h(g)} \\
\end{align}
$$

The $home$ covariate will be constant through time and for all teams.
The score for a team is determined by the "attack" ($att$) and "defensibility" ($def$) of the two teams.

The attack and defensibility for each team $t$ come from common distributions.

$$
\begin{align}
att_t &\sim \mathcal{N}(\mu_{att}, \tau_{att}) \\
def_t &\sim \mathcal{N}(\mu_{def}, \tau_{def}) \\
\end{align}
$$

To help with sampling, the actual model will be reparamaterized using "non-centered parameterization." Here is a good link for learning about this technique: ["Why hierarchical models are awesome, tricky, and Bayesian"](https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/#Reparameterization).

```python
df = df_all[["home_team", "away_team", "home_score", "away_score"]].reset_index(
    drop=True
)

for col in ["home_team", "away_team"]:
    df[col] = pd.Categorical(df[col].values, ordered=True)

home_team = df.home_team.cat.codes.to_numpy()
away_team = df.away_team.cat.codes.to_numpy()

num_teams = len(np.unique(home_team))
num_games = df.shape[0]
```

```python
with pm.Model() as model:

    # Global model parameters.
    home = pm.Normal("home", 1, 10)
    τ_att = pm.HalfStudentT("τ_att", nu=3, sigma=2.5)
    τ_def = pm.HalfStudentT("τ_def", nu=3, sigma=2.5)
    α = pm.Normal("α", 0, 5)  # shared intercept

    # Team-specific model parameters.
    att_star_g = pm.Normal("att_star_g", 0, τ_att, shape=num_teams)
    def_star_g = pm.Normal("def_star_g", 0, τ_def, shape=num_teams)

    att_g = pm.Deterministic("att_g", att_star_g - tt.mean(att_star_g))
    def_g = pm.Deterministic("def_g", def_star_g - tt.mean(def_star_g))
    θ_home = tt.exp(α + home + att_g[home_team] + def_g[away_team])
    θ_away = tt.exp(α + att_g[away_team] + def_g[home_team])

    # Likelihood of observed data.
    home_points = pm.Poisson("home_points", mu=θ_home, observed=df.home_score)
    away_points = pm.Poisson("away_points", mu=θ_away, observed=df.away_score)
```

```python
pm.model_to_graphviz(model)
```

![svg](013_rugby-predictions_files/013_rugby-predictions_14_0.svg)

```python
with model:
    trace = pm.sample(1000, tune=2000, chains=3, cores=3)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (3 chains in 3 jobs)
    NUTS: [def_star_g, att_star_g, α, τ_def, τ_att, home]

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
  <progress value='9000' class='' max='9000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [9000/9000 00:37<00:00 Sampling 3 chains, 0 divergences]
</div>

    Sampling 3 chains for 2_000 tune and 1_000 draw iterations (6_000 + 3_000 draws total) took 53 seconds.

```python
az_model = az.from_pymc3(trace=trace, model=model)
```

```python
az.plot_trace(az_model)
plt.show()
```

![png](013_rugby-predictions_files/013_rugby-predictions_17_0.png)

As in an good statistical workflow, let us check some evaluation metrics to ensure the NUTS sampler converged.

```python
bfmi = np.max(az.bfmi(az_model))
max_gr = np.max(list(np.max(gr_stats) for gr_stats in az.rhat(az_model).values()))

az.plot_energy(az_model, figsize=(6, 4)).set_title(
    f"BFMI = {bfmi:.3f} | Gelman-Rubin = {max_gr:.3f}"
)
plt.show()
```

![png](013_rugby-predictions_files/013_rugby-predictions_19_0.png)

## Results

```python

```
