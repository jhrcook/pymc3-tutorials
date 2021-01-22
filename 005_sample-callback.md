# Sample callback

[Tutorial](https://docs.pymc.io/notebooks/sampling_callback.html)

```python
from typing import Dict, List

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import pymc3 as pm

%config InlineBackend.figure_format = "retina"
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")
gg.theme_set(gg.theme_minimal)
```

    plotnine.themes.theme_minimal.theme_minimal

**callback**: a function which gets called for every sample from the trace of a chain

In PyMC3, the function is called with a single trace and current draw as arguments. Some usecaes:

- stopping sampling when a number of effective samples or Rhat is reached
- stopping sampling after too many divergences
- logging metrics to external tools (e.g. Tensorboard)

Below is a callback that stops sampling after 100 samples

```python
def my_callback(
    trace: pm.backends.ndarray.NDArray, draw: pm.parallel_sampling.Draw
) -> None:
    if len(trace) >= 100:
        raise KeyboardInterrupt()


X = np.arange(1, 6)
y = X * 2 + np.random.randn(len(X))

with pm.Model() as model:
    intercept = pm.Normal("intercept", 0, 10)
    slope = pm.Normal("slope", 0, 10)

    mean = intercept + slope * X
    error = pm.HalfCauchy("error", 1)

    obs = pm.Normal("obs", mean, error, observed=y)

    trace = pm.sample(500, tune=0, chains=1, callback=my_callback)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Sequential sampling (1 chains in 1 job)
    NUTS: [error, slope, intercept]

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
  <progress value='86' class='' max='500' style='width:300px; height:20px; vertical-align: middle;'></progress>
  17.20% [86/500 00:00<00:02 Sampling chain 0, 1 divergences]
</div>

    Sampling 1 chain for 0 tune and 100 draw iterations (0 + 100 draws total) took 1 seconds.
    There was 1 divergence after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.6926582589473618, but should be close to 0.8. Try to increase the number of tuning steps.
    Only one chain was sampled, this makes it impossible to run some convergence checks

Each call back is only given a single chain.
Thus, need to create a class in order to do calculations over multiple chains at once.
Below is an example of stoping a chain after the Rhat is low enough to indicate convergence.

```python
class MyRhatCheckingCallback:
    def __init__(self, every: int = 1000, max_rhat: float = 1.05) -> None:
        self.every = every
        self.max_rhat = max_rhat
        self.traces: Dict[int, pm.backends.ndarray.NDArray] = {}

    def __call__(
        self, trace: pm.backends.ndarray.NDArray, draw: pm.parallel_sampling.Draw
    ) -> None:
        if draw.tuning:
            return

        self.traces[draw.chain] = trace
        if len(trace) % self.every == 0:
            traces = self.get_trimmed_traces()
            multitrace = pm.backends.base.MultiTrace(traces)
            x = pm.stats.rhat(multitrace)
            if x.to_array().max() < self.max_rhat:
                raise KeyboardInterrupt()

    def get_trimmed_traces(self) -> List[pm.backends.ndarray.NDArray]:
        traces = list(self.traces.values())
        trace_min_length = np.min([len(t) for t in traces])
        return [t[:trace_min_length] for t in traces]


with model:
    trace = pm.sample(
        tune=1000, draws=1000000, callback=MyRhatCheckingCallback(), chains=2, cores=2
    )
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [error, slope, intercept]

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
  <progress value='5670' class='' max='2002000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  0.28% [5670/2002000 00:10<1:03:48 Sampling 2 chains, 25 divergences]
</div>

    Sampling 1 chain for 1_000 tune and 3_000 draw iterations (1_000 + 3_000 draws total) took 21 seconds.
    Only one chain was sampled, this makes it impossible to run some convergence checks

---

```python
%load_ext watermark
%watermark -n -u -v -iv -w
```

    Last updated: Fri Jan 22 2021

    Python implementation: CPython
    Python version       : 3.9.1
    IPython version      : 7.19.0

    arviz     : 0.11.0
    pandas    : 1.2.0
    pymc3     : 3.9.3
    numpy     : 1.19.5
    matplotlib: 3.3.3
    plotnine  : 0.7.1

    Watermark: 2.1.0
