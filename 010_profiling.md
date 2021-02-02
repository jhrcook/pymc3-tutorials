# Profiling

**[Original tutorial](https://docs.pymc.io/notebooks/profiling.html)**

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

PyMC3 has wrapped profiling tools from Theano into `model.profile()`.
This function returns a `ProfileStats` object with information about the underlying Theano operations.

Below is an example of profiling the likelihood and gradient for the stochastic volatility example.

```python
returns = pd.read_csv(pm.get_data("SP500.csv"), index_col=0, parse_dates=True)
returns.head()
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
      <th>Close</th>
      <th>change</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2008-05-02</th>
      <td>1413.900024</td>
      <td>0.003230</td>
    </tr>
    <tr>
      <th>2008-05-05</th>
      <td>1407.489990</td>
      <td>-0.004544</td>
    </tr>
    <tr>
      <th>2008-05-06</th>
      <td>1418.260010</td>
      <td>0.007623</td>
    </tr>
    <tr>
      <th>2008-05-07</th>
      <td>1392.569946</td>
      <td>-0.018280</td>
    </tr>
    <tr>
      <th>2008-05-08</th>
      <td>1397.680054</td>
      <td>0.003663</td>
    </tr>
  </tbody>
</table>
</div>

```python
with pm.Model() as model:
    sigma = pm.Exponential("sigma", 1.0 / 0.02, testval=0.1)
    nu = pm.Exponential("nu", 1.0 / 10.0)
    s = pm.GaussianRandomWalk("s", sigma ** -2, shape=returns.shape[0])
    r = pm.StudentT("r", nu, lam=np.exp(-2 * s), observed=returns["change"])
```

To profile, call the `profile()` function and summarise the return values.

```python
model.profile(model.logpt).summary()
```

    Function profiling
    ==================
      Message: /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/pymc3/model.py:1191
      Time in 1000 calls to Function.__call__: 1.321383e-01s
      Time in Function.fn.__call__: 1.078756e-01s (81.638%)
      Time in thunks: 9.906149e-02s (74.968%)
      Total compile time: 1.016328e+00s
        Number of Apply nodes: 22
        Theano Optimizer time: 8.974490e-01s
           Theano validate time: 2.188683e-03s
        Theano Linker time (includes C, CUDA code generation/compiling): 5.971813e-02s
           Import time 3.460383e-02s
           Node make_thunk time 5.846930e-02s
               Node Elemwise{Composite{log((i0 * i1))}}(TensorConstant{(1,) of 0...4309189535}, Elemwise{Composite{inv(sqr(i0))}}[(0, 0)].0) time 7.901907e-03s
               Node Elemwise{Composite{Switch(Cast{int8}((GT(Composite{exp((i0 * i1))}(i0, i1), i2) * i3 * GT(inv(sqrt(Composite{exp((i0 * i1))}(i0, i1))), i2))), (((i4 + (i5 * log(((i6 * Composite{exp((i0 * i1))}(i0, i1)) / i7)))) - i8) - (i5 * i9 * log1p(((Composite{exp((i0 * i1))}(i0, i1) * i10) / i7)))), i11)}}(TensorConstant{(1,) of -2.0}, s, TensorConstant{(1,) of 0}, Elemwise{gt,no_inplace}.0, Elemwise{Composite{scalar_gammaln((i0 * i1))}}.0, TensorConstant{(1,) of 0.5}, TensorConstant{(1,) of 0...8861837907}, InplaceDimShuffle{x}.0, Elemwise{Composite{scalar_gammaln((i0 * i1))}}.0, Elemwise{add,no_inplace}.0, TensorConstant{[1.0435322..54666e-07]}, TensorConstant{(1,) of -inf}) time 5.974054e-03s
               Node Elemwise{Composite{scalar_gammaln((i0 * i1))}}(TensorConstant{(1,) of 0.5}, InplaceDimShuffle{x}.0) time 4.871845e-03s
               Node Elemwise{Composite{(Switch(Cast{int8}(GE(i0, i1)), (i2 - (i3 * i0)), i4) + i5)}}[(0, 0)](nu, TensorConstant{0}, TensorConstant{-2.3025850929940455}, TensorConstant{0.1}, TensorConstant{-inf}, nu_log__) time 4.812956e-03s
               Node Elemwise{Composite{inv(sqrt(inv(sqr(i0))))}}[(0, 0)](InplaceDimShuffle{x}.0) time 4.173040e-03s

    Time in all call to theano.grad() 0.000000e+00s
    Time since theano import 9.624s
    Class
    ---
    <% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Class name>
      88.6%    88.6%       0.088s       6.27e-06s     C    14000      14   theano.tensor.elemwise.Elemwise
       8.4%    96.9%       0.008s       2.76e-06s     C     3000       3   theano.tensor.elemwise.Sum
       1.7%    98.7%       0.002s       8.56e-07s     C     2000       2   theano.tensor.elemwise.DimShuffle
       0.9%    99.6%       0.001s       4.43e-07s     C     2000       2   theano.tensor.subtensor.Subtensor
       0.4%   100.0%       0.000s       4.26e-07s     C     1000       1   theano.tensor.opt.MakeVector
       ... (remaining 0 Classes account for   0.00%(0.00s) of the runtime)

    Ops
    ---
    <% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Op name>
      79.0%    79.0%       0.078s       7.83e-05s     C     1000        1   Elemwise{Composite{Switch(Cast{int8}((GT(Composite{exp((i0 * i1))}(i0, i1), i2) * i3 * GT(inv(sqrt(Composite{exp((i0 * i1))}(i0, i1))), i2))), (((i4 + (i5 * log(((i6 * Composite{exp((i0 * i1))}(i0, i1)) / i7)))) - i8) - (i5 * i9 * log1p(((Composite{exp((i0 * i1))}(i0, i1) * i10) / i7)))), i11)}}
       8.4%    87.4%       0.008s       2.76e-06s     C     3000        3   Sum{acc_dtype=float64}
       5.6%    93.0%       0.006s       5.59e-06s     C     1000        1   Elemwise{Composite{Switch(i0, (i1 * ((-(i2 * sqr((i3 - i4)))) + i5)), i6)}}
       1.7%    94.7%       0.002s       8.56e-07s     C     2000        2   InplaceDimShuffle{x}
       0.7%    95.5%       0.001s       3.53e-07s     C     2000        2   Elemwise{exp,no_inplace}
       0.7%    96.1%       0.001s       3.40e-07s     C     2000        2   Elemwise{Composite{scalar_gammaln((i0 * i1))}}
       0.5%    96.7%       0.001s       5.12e-07s     C     1000        1   Subtensor{int64::}
       0.4%    97.1%       0.000s       4.26e-07s     C     1000        1   MakeVector{dtype='float64'}
       0.4%    97.5%       0.000s       4.10e-07s     C     1000        1   Elemwise{Composite{(Switch(Cast{int8}(GE(i0, i1)), (i2 - (i3 * i0)), i4) + i5)}}
       0.4%    97.9%       0.000s       3.89e-07s     C     1000        1   Elemwise{Composite{log((i0 * i1))}}
       0.4%    98.3%       0.000s       3.75e-07s     C     1000        1   Subtensor{:int64:}
       0.4%    98.7%       0.000s       3.68e-07s     C     1000        1   Elemwise{add,no_inplace}
       0.4%    99.0%       0.000s       3.68e-07s     C     1000        1   Elemwise{Composite{(Switch(Cast{int8}(GE(i0, i1)), (i2 - (i3 * i0)), i4) + i5)}}[(0, 0)]
       0.3%    99.3%       0.000s       3.21e-07s     C     1000        1   Elemwise{gt,no_inplace}
       0.3%    99.7%       0.000s       3.01e-07s     C     1000        1   Elemwise{Composite{Cast{int8}(GT(i0, i1))}}
       0.2%    99.8%       0.000s       1.89e-07s     C     1000        1   Elemwise{Composite{inv(sqrt(inv(sqr(i0))))}}[(0, 0)]
       0.2%   100.0%       0.000s       1.58e-07s     C     1000        1   Elemwise{Composite{inv(sqr(i0))}}[(0, 0)]
       ... (remaining 0 Ops account for   0.00%(0.00s) of the runtime)

    Apply
    ------
    <% time> <sum %> <apply time> <time per call> <#call> <id> <Apply name>
      79.0%    79.0%       0.078s       7.83e-05s   1000    14   Elemwise{Composite{Switch(Cast{int8}((GT(Composite{exp((i0 * i1))}(i0, i1), i2) * i3 * GT(inv(sqrt(Composite{exp((i0 * i1))}(i0, i1))), i2))), (((i4 + (i5 * log(((i6 * Composite{exp((i0 * i1))}(i0, i1)) / i7)))) - i8) - (i5 * i9 * log1p(((Composite{exp((i0 * i1))}(i0, i1) * i10) / i7)))), i11)}}(TensorConstant{(1,) of -2.0}, s, TensorConstant{(1,) of 0}, Elemwise{gt,no_inplace}.0, Elemwise{Composite{scalar_gammaln((i0 * i1))}}.0, TensorConstant{(1,)
       5.6%    84.7%       0.006s       5.59e-06s   1000    18   Elemwise{Composite{Switch(i0, (i1 * ((-(i2 * sqr((i3 - i4)))) + i5)), i6)}}(Elemwise{Composite{Cast{int8}(GT(i0, i1))}}.0, TensorConstant{(1,) of 0.5}, Elemwise{Composite{inv(sqr(i0))}}[(0, 0)].0, Subtensor{int64::}.0, Subtensor{:int64:}.0, Elemwise{Composite{log((i0 * i1))}}.0, TensorConstant{(1,) of -inf})
       4.2%    88.9%       0.004s       4.19e-06s   1000    17   Sum{acc_dtype=float64}(Elemwise{Composite{Switch(Cast{int8}((GT(Composite{exp((i0 * i1))}(i0, i1), i2) * i3 * GT(inv(sqrt(Composite{exp((i0 * i1))}(i0, i1))), i2))), (((i4 + (i5 * log(((i6 * Composite{exp((i0 * i1))}(i0, i1)) / i7)))) - i8) - (i5 * i9 * log1p(((Composite{exp((i0 * i1))}(i0, i1) * i10) / i7)))), i11)}}.0)
       4.0%    92.9%       0.004s       3.94e-06s   1000    19   Sum{acc_dtype=float64}(Elemwise{Composite{Switch(i0, (i1 * ((-(i2 * sqr((i3 - i4)))) + i5)), i6)}}.0)
       0.9%    93.8%       0.001s       9.30e-07s   1000     4   InplaceDimShuffle{x}(nu)
       0.8%    94.6%       0.001s       7.82e-07s   1000     6   InplaceDimShuffle{x}(sigma)
       0.5%    95.1%       0.001s       5.21e-07s   1000     1   Elemwise{exp,no_inplace}(sigma_log__)
       0.5%    95.6%       0.001s       5.12e-07s   1000     3   Subtensor{int64::}(s, Constant{1})
       0.4%    96.1%       0.000s       4.26e-07s   1000    20   MakeVector{dtype='float64'}(__logp_sigma_log__, __logp_nu_log__, __logp_s, __logp_r)
       0.4%    96.5%       0.000s       4.16e-07s   1000     8   Elemwise{Composite{scalar_gammaln((i0 * i1))}}(TensorConstant{(1,) of 0.5}, InplaceDimShuffle{x}.0)
       0.4%    96.9%       0.000s       4.10e-07s   1000     5   Elemwise{Composite{(Switch(Cast{int8}(GE(i0, i1)), (i2 - (i3 * i0)), i4) + i5)}}(sigma, TensorConstant{0}, TensorConstant{3.912023005428146}, TensorConstant{50.0}, TensorConstant{-inf}, sigma_log__)
       0.4%    97.3%       0.000s       3.89e-07s   1000    16   Elemwise{Composite{log((i0 * i1))}}(TensorConstant{(1,) of 0...4309189535}, Elemwise{Composite{inv(sqr(i0))}}[(0, 0)].0)
       0.4%    97.7%       0.000s       3.75e-07s   1000     2   Subtensor{:int64:}(s, Constant{-1})
       0.4%    98.0%       0.000s       3.68e-07s   1000    15   Elemwise{Composite{(Switch(Cast{int8}(GE(i0, i1)), (i2 - (i3 * i0)), i4) + i5)}}[(0, 0)](nu, TensorConstant{0}, TensorConstant{-2.3025850929940455}, TensorConstant{0.1}, TensorConstant{-inf}, nu_log__)
       0.4%    98.4%       0.000s       3.68e-07s   1000     7   Elemwise{add,no_inplace}(TensorConstant{(1,) of 1.0}, InplaceDimShuffle{x}.0)
       0.3%    98.7%       0.000s       3.21e-07s   1000     9   Elemwise{gt,no_inplace}(InplaceDimShuffle{x}.0, TensorConstant{(1,) of 0})
       0.3%    99.0%       0.000s       3.01e-07s   1000    12   Elemwise{Composite{Cast{int8}(GT(i0, i1))}}(Elemwise{Composite{inv(sqrt(inv(sqr(i0))))}}[(0, 0)].0, TensorConstant{(1,) of 0})
       0.3%    99.3%       0.000s       2.64e-07s   1000    11   Elemwise{Composite{scalar_gammaln((i0 * i1))}}(TensorConstant{(1,) of 0.5}, Elemwise{add,no_inplace}.0)
       0.2%    99.5%       0.000s       1.89e-07s   1000    10   Elemwise{Composite{inv(sqrt(inv(sqr(i0))))}}[(0, 0)](InplaceDimShuffle{x}.0)
       0.2%    99.7%       0.000s       1.86e-07s   1000     0   Elemwise{exp,no_inplace}(nu_log__)
       ... (remaining 2 Apply instances account for 0.32%(0.00s) of the runtime)

    Here are tips to potentially make your code run faster
                     (if you think of new ones, suggest them on the mailing list).
                     Test them first, as they are not guaranteed to always provide a speedup.
      - Try the Theano flag floatX=float32
    We don't know if amdlibm will accelerate this scalar op. scalar_gammaln
    We don't know if amdlibm will accelerate this scalar op. scalar_gammaln
      - Try installing amdlibm and set the Theano flag lib.amdlibm=True. This speeds up only some Elemwise operation.

We can also profile the gradient call `dlogp / dx`.

```python
model.profile(pm.gradient(model.logpt, model.vars)).summary()
```

    Function profiling
    ==================
      Message: /usr/local/Caskroom/miniconda/base/envs/pymc3-tutorials/lib/python3.9/site-packages/pymc3/model.py:1191
      Time in 1000 calls to Function.__call__: 1.813309e-01s
      Time in Function.fn.__call__: 1.566050e-01s (86.364%)
      Time in thunks: 1.386719e-01s (76.474%)
      Total compile time: 2.309332e+00s
        Number of Apply nodes: 47
        Theano Optimizer time: 1.987331e+00s
           Theano validate time: 6.048441e-03s
        Theano Linker time (includes C, CUDA code generation/compiling): 2.872701e-01s
           Import time 6.456828e-02s
           Node make_thunk time 2.851560e-01s
               Node Elemwise{Composite{Switch(i0, (i1 * (i2 - i3)), i4)}}(Elemwise{Composite{Cast{int8}(GT(i0, i1))}}.0, InplaceDimShuffle{x}.0, Subtensor{int64::}.0, Subtensor{:int64:}.0, TensorConstant{(1,) of 0}) time 1.713119e-01s
               Node Elemwise{Composite{Switch(i0, (i1 * (i2 + ((i3 * i4 * i5 * i6) / i7))), i8)}}[(0, 6)](Elemwise{Composite{Cast{int8}((GT(i0, i1) * i2 * GT(inv(sqrt(i0)), i1)))}}.0, TensorConstant{(1,) of -2.0}, TensorConstant{(1,) of 0.5}, TensorConstant{(1,) of -0.5}, Elemwise{add,no_inplace}.0, TensorConstant{[1.0435322..54666e-07]}, Elemwise{Composite{exp((i0 * i1))}}.0, Elemwise{Add}[(0, 1)].0, TensorConstant{(1,) of 0}) time 6.316185e-03s
               Node Elemwise{Composite{Switch(i0, (-log1p((i1 / i2))), i3)}}(Elemwise{Composite{Cast{int8}((GT(i0, i1) * i2 * GT(inv(sqrt(i0)), i1)))}}.0, Elemwise{mul,no_inplace}.0, InplaceDimShuffle{x}.0, TensorConstant{(1,) of 0}) time 6.295919e-03s
               Node Elemwise{Composite{Switch(i0, ((i1 * i2 * i3 * i4) / i5), i6)}}(Elemwise{Composite{Cast{int8}((GT(i0, i1) * i2 * GT(inv(sqrt(i0)), i1)))}}.0, TensorConstant{(1,) of 0.5}, Elemwise{add,no_inplace}.0, Elemwise{Composite{exp((i0 * i1))}}.0, TensorConstant{[1.0435322..54666e-07]}, Elemwise{Add}[(0, 1)].0, TensorConstant{(1,) of 0}) time 6.274700e-03s
               Node Elemwise{Composite{Switch(i0, (i1 * i2 * (i3 - i4)), i5)}}(Elemwise{Composite{Cast{int8}(GT(i0, i1))}}.0, TensorConstant{(1,) of -1.0}, InplaceDimShuffle{x}.0, Subtensor{int64::}.0, Subtensor{:int64:}.0, TensorConstant{(1,) of 0}) time 6.027937e-03s

    Time in all call to theano.grad() 1.172364e+00s
    Time since theano import 13.726s
    Class
    ---
    <% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Class name>
      65.7%    65.7%       0.091s       3.65e-06s     C    25000      25   theano.tensor.elemwise.Elemwise
      20.1%    85.9%       0.028s       3.99e-06s     C     7000       7   theano.tensor.elemwise.Sum
       6.5%    92.4%       0.009s       4.53e-06s     C     2000       2   theano.tensor.subtensor.IncSubtensor
       2.7%    95.1%       0.004s       3.73e-06s     C     1000       1   theano.tensor.basic.Join
       2.0%    97.1%       0.003s       7.10e-07s     C     4000       4   theano.tensor.elemwise.DimShuffle
       1.4%    98.5%       0.002s       1.88e-06s     C     1000       1   theano.tensor.basic.Alloc
       0.6%    99.1%       0.001s       4.37e-07s     C     2000       2   theano.tensor.subtensor.Subtensor
       0.5%    99.6%       0.001s       3.72e-07s     C     2000       2   theano.tensor.basic.Reshape
       0.2%    99.8%       0.000s       2.43e-07s     C     1000       1   theano.compile.ops.Shape_i
       0.2%   100.0%       0.000s       1.21e-07s     C     2000       2   theano.compile.ops.Rebroadcast
       ... (remaining 0 Classes account for   0.00%(0.00s) of the runtime)

    Ops
    ---
    <% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Op name>
      25.6%    25.6%       0.036s       3.55e-05s     C     1000        1   Elemwise{Composite{Switch(i0, (-log1p((i1 / i2))), i3)}}
      20.1%    45.7%       0.028s       3.99e-06s     C     7000        7   Sum{acc_dtype=float64}
       7.5%    53.3%       0.010s       1.05e-05s     C     1000        1   Elemwise{Composite{exp((i0 * i1))}}
       6.6%    59.9%       0.009s       9.12e-06s     C     1000        1   Elemwise{Composite{Cast{int8}((GT(i0, i1) * i2 * GT(inv(sqrt(i0)), i1)))}}
       5.8%    65.6%       0.008s       2.00e-06s     C     4000        4   Elemwise{switch,no_inplace}
       4.8%    70.4%       0.007s       6.67e-06s     C     1000        1   Elemwise{Composite{Switch(i0, (i1 * (i2 + ((i3 * i4 * i5 * i6) / i7))), i8)}}[(0, 6)]
       4.1%    74.6%       0.006s       5.72e-06s     C     1000        1   IncSubtensor{InplaceInc;int64::}
       3.4%    78.0%       0.005s       4.77e-06s     C     1000        1   Elemwise{Composite{Switch(i0, ((i1 * i2 * i3 * i4) / i5), i6)}}
       2.7%    80.7%       0.004s       3.76e-06s     C     1000        1   Elemwise{Composite{Switch(i0, (i1 * i2 * (i3 - i4)), i5)}}
       2.7%    83.4%       0.004s       3.73e-06s     C     1000        1   Join
       2.4%    85.8%       0.003s       3.33e-06s     C     1000        1   IncSubtensor{InplaceInc;:int64:}
       2.3%    88.1%       0.003s       3.12e-06s     C     1000        1   Elemwise{Composite{Switch(i0, (i1 * sqr((i2 - i3))), i4)}}
       2.2%    90.3%       0.003s       3.10e-06s     C     1000        1   Elemwise{Composite{Switch(i0, (i1 * (i2 - i3)), i4)}}
       2.0%    92.3%       0.003s       7.10e-07s     C     4000        4   InplaceDimShuffle{x}
       1.4%    93.7%       0.002s       1.89e-06s     C     1000        1   Elemwise{mul,no_inplace}
       1.4%    95.1%       0.002s       1.88e-06s     C     1000        1   Alloc
       0.5%    95.6%       0.001s       3.72e-07s     C     2000        2   Reshape{1}
       0.5%    96.1%       0.001s       7.36e-07s     C     1000        1   Elemwise{Composite{(Switch(Cast{int8}(GE(i0, i1)), (i2 * i0), i1) + i3 + (i4 * i5 * psi((i4 * (i6 + i0))) * i0) + (i7 * i8) + (i4 * i9 * psi((i4 * i0)) * i0) + (i4 * i10 * i0) + i11)}}[(0, 0)]
       0.5%    96.7%       0.001s       3.64e-07s     C     2000        2   Elemwise{exp,no_inplace}
       0.5%    97.2%       0.001s       7.07e-07s     C     1000        1   Elemwise{Composite{(Switch(Cast{int8}(GE(i0, i1)), (i2 * i0), i1) + i3 + (i4 * (((i5 * i6 * Composite{inv(Composite{(sqr(i0) * i0)}(i0))}(i7)) / i8) - (i9 * Composite{inv(Composite{(sqr(i0) * i0)}(i0))}(i7))) * (i10 ** i11) * inv(Composite{(sqr(i0) * i0)}(i0)) * i0))}}[(0, 0)]
       ... (remaining 11 Ops account for   2.83%(0.00s) of the runtime)

    Apply
    ------
    <% time> <sum %> <apply time> <time per call> <#call> <id> <Apply name>
      25.6%    25.6%       0.036s       3.55e-05s   1000    19   Elemwise{Composite{Switch(i0, (-log1p((i1 / i2))), i3)}}(Elemwise{Composite{Cast{int8}((GT(i0, i1) * i2 * GT(inv(sqrt(i0)), i1)))}}.0, Elemwise{mul,no_inplace}.0, InplaceDimShuffle{x}.0, TensorConstant{(1,) of 0})
       7.5%    33.2%       0.010s       1.05e-05s   1000     5   Elemwise{Composite{exp((i0 * i1))}}(TensorConstant{(1,) of -2.0}, s)
       6.6%    39.7%       0.009s       9.12e-06s   1000    15   Elemwise{Composite{Cast{int8}((GT(i0, i1) * i2 * GT(inv(sqrt(i0)), i1)))}}(Elemwise{Composite{exp((i0 * i1))}}.0, TensorConstant{(1,) of 0}, Elemwise{gt,no_inplace}.0)
       4.8%    44.5%       0.007s       6.67e-06s   1000    35   Elemwise{Composite{Switch(i0, (i1 * (i2 + ((i3 * i4 * i5 * i6) / i7))), i8)}}[(0, 6)](Elemwise{Composite{Cast{int8}((GT(i0, i1) * i2 * GT(inv(sqrt(i0)), i1)))}}.0, TensorConstant{(1,) of -2.0}, TensorConstant{(1,) of 0.5}, TensorConstant{(1,) of -0.5}, Elemwise{add,no_inplace}.0, TensorConstant{[1.0435322..54666e-07]}, Elemwise{Composite{exp((i0 * i1))}}.0, Elemwise{Add}[(0, 1)].0, TensorConstant{(1,) of 0})
       4.1%    48.7%       0.006s       5.72e-06s   1000    38   IncSubtensor{InplaceInc;int64::}(Elemwise{Composite{Switch(i0, (i1 * (i2 + ((i3 * i4 * i5 * i6) / i7))), i8)}}[(0, 6)].0, Elemwise{Composite{Switch(i0, (i1 * i2 * (i3 - i4)), i5)}}.0, Constant{1})
       3.4%    52.1%       0.005s       4.77e-06s   1000    32   Elemwise{Composite{Switch(i0, ((i1 * i2 * i3 * i4) / i5), i6)}}(Elemwise{Composite{Cast{int8}((GT(i0, i1) * i2 * GT(inv(sqrt(i0)), i1)))}}.0, TensorConstant{(1,) of 0.5}, Elemwise{add,no_inplace}.0, Elemwise{Composite{exp((i0 * i1))}}.0, TensorConstant{[1.0435322..54666e-07]}, Elemwise{Add}[(0, 1)].0, TensorConstant{(1,) of 0})
       3.0%    55.1%       0.004s       4.12e-06s   1000    37   Sum{acc_dtype=float64}(Alloc.0)
       3.0%    58.0%       0.004s       4.11e-06s   1000    36   Sum{acc_dtype=float64}(Elemwise{Composite{Switch(i0, ((i1 * i2 * i3 * i4) / i5), i6)}}.0)
       2.9%    61.0%       0.004s       4.05e-06s   1000    28   Sum{acc_dtype=float64}(Elemwise{Composite{Switch(i0, (-log1p((i1 / i2))), i3)}}.0)
       2.9%    63.8%       0.004s       3.99e-06s   1000    33   Sum{acc_dtype=float64}(Elemwise{Composite{Switch(i0, (i1 * sqr((i2 - i3))), i4)}}.0)
       2.8%    66.7%       0.004s       3.91e-06s   1000    30   Sum{acc_dtype=float64}(Elemwise{switch,no_inplace}.0)
       2.8%    69.5%       0.004s       3.89e-06s   1000    31   Sum{acc_dtype=float64}(Elemwise{Switch}.0)
       2.8%    72.2%       0.004s       3.86e-06s   1000    29   Sum{acc_dtype=float64}(Elemwise{switch,no_inplace}.0)
       2.7%    75.0%       0.004s       3.76e-06s   1000    25   Elemwise{Composite{Switch(i0, (i1 * i2 * (i3 - i4)), i5)}}(Elemwise{Composite{Cast{int8}(GT(i0, i1))}}.0, TensorConstant{(1,) of -1.0}, InplaceDimShuffle{x}.0, Subtensor{int64::}.0, Subtensor{:int64:}.0, TensorConstant{(1,) of 0})
       2.7%    77.7%       0.004s       3.73e-06s   1000    46   Join(TensorConstant{0}, Rebroadcast{1}.0, Rebroadcast{1}.0, (d__logp/ds))
       2.6%    80.2%       0.004s       3.55e-06s   1000    22   Elemwise{Switch}(Elemwise{Composite{Cast{int8}((GT(i0, i1) * i2 * GT(inv(sqrt(i0)), i1)))}}.0, TensorConstant{(1,) of 1.0}, TensorConstant{(1,) of 0.0})
       2.4%    82.6%       0.003s       3.33e-06s   1000    41   IncSubtensor{InplaceInc;:int64:}(IncSubtensor{InplaceInc;int64::}.0, Elemwise{Composite{Switch(i0, (i1 * (i2 - i3)), i4)}}.0, Constant{-1})
       2.3%    84.9%       0.003s       3.12e-06s   1000    26   Elemwise{Composite{Switch(i0, (i1 * sqr((i2 - i3))), i4)}}(Elemwise{Composite{Cast{int8}(GT(i0, i1))}}.0, TensorConstant{(1,) of 0.5}, Subtensor{int64::}.0, Subtensor{:int64:}.0, TensorConstant{(1,) of 0})
       2.2%    87.1%       0.003s       3.10e-06s   1000    24   Elemwise{Composite{Switch(i0, (i1 * (i2 - i3)), i4)}}(Elemwise{Composite{Cast{int8}(GT(i0, i1))}}.0, InplaceDimShuffle{x}.0, Subtensor{int64::}.0, Subtensor{:int64:}.0, TensorConstant{(1,) of 0})
       1.5%    88.6%       0.002s       2.13e-06s   1000    21   Elemwise{switch,no_inplace}(Elemwise{Composite{Cast{int8}((GT(i0, i1) * i2 * GT(inv(sqrt(i0)), i1)))}}.0, TensorConstant{(1,) of -0..4309189535}, TensorConstant{(1,) of 0})
       ... (remaining 27 Apply instances account for 11.37%(0.02s) of the runtime)

    Here are tips to potentially make your code run faster
                     (if you think of new ones, suggest them on the mailing list).
                     Test them first, as they are not guaranteed to always provide a speedup.
      - Try the Theano flag floatX=float32
    We don't know if amdlibm will accelerate this scalar op. psi
    We don't know if amdlibm will accelerate this scalar op. psi
      - Try installing amdlibm and set the Theano flag lib.amdlibm=True. This speeds up only some Elemwise operation.

---

```python
%load_ext watermark
%watermark -n -u -v -iv -w
```

    Last updated: Tue Feb 02 2021

    Python implementation: CPython
    Python version       : 3.9.1
    IPython version      : 7.20.0

    pymc3     : 3.9.3
    plotnine  : 0.7.1
    arviz     : 0.11.0
    numpy     : 1.20.0
    matplotlib: 3.3.4
    pandas    : 1.2.1

    Watermark: 2.1.0
