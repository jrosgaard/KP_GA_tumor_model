# Dixon simulation and analysis

Run the no-treatment and alpha-sweep cells in `main_dixon.ipynb`, then run
`Analysis_Dixon.ipynb`. The parameter mapping
`base_params` is the place to change model coefficients. Those coefficients
are already nondimensional; do not nondimensionalize them a second time.

The reference is [Dixon et al. (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11193449/),
equation (2), section 3.1, and Table 5. The simulation
uses the Table 5 nondimensional coefficients, with `s=0` and `mu_2=0.73`
for the no-treatment simulation.

## Physical scales

- Physical time `t` is in days; `tau = lambda_tumor * t`.
- `lambda_tumor = 0.18 day^-1` is independent of the observation duration.
- A 4,000-day observation window therefore covers `tau=0..720`.
- `K=1e9` is the cell scale and, numerically, the paper's IL-2 scale in IU.
- Every physical state is restored using its nondimensional state times `K`.
- The treatment coefficient `s` is nondimensional; `s_bar = s*K*lambda_tumor`
  is the physical IL-2 source rate in IU/day.

The original single-run IL-2 restoration multiplied by its initial value even
though it had been divided by `K`; the new simulator uses `K` consistently.
The physical initial populations for the baseline are retained as
`[1e5, 1e5, 1e2, 1e5]`, ordered `[C1, C2, I, T]`. With the paper's `K`, their
nondimensional values are `[1e-4, 1e-4, 1e-7, 1e-4]`.

## Integration and output

The structure follows KP: equations live in `Model/Dixon_model.py`, numerical
integration lives in `Model/integration.py`, and the notebook owns parameters,
state arrays, the time-stepping loop, dimensionalization, and CSV output.
There is no separate simulation module for either model.

`dixon_integrate.integrate(state, params, t_span, method=...)` advances one
observation interval, just like `kp_integrate.integrate`. Dixon additionally
supports SciPy Radau for its fast IL-2 decay. Ordinary calls still take and
return `[C1, C2, I, T]`.

For underflow-prone trajectories, the notebook enables `log_cells=True` and
carries `[log(C1), log(C2), I, log(T)]` between steps. Conversion methods on
the integrator keep this numerical detail out of the physical state arrays.
`log10_T` records the saved logarithmic tumor size. An exactly zero initial
population stays zero. Change treatment parameters between intervals to
simulate piecewise inputs; `log_cell_integration=False` selects ordinary
state integration.

Sweep output location: `Output_data/dixon_alpha_sweep_no_treatment_v2/`.
The single-run plot uses arrays in memory. Analysis selects an existing alpha
sweep CSV, and extracted features stay in memory for FFT and clustering.

Previous files are retained. Each new CSV records the coefficients, scales,
and simulation version. The sweep skips an existing result only if its
settings and observation times agree; a settings mismatch requires choosing
a new output directory or explicitly setting `overwrite_existing=True`.

The alpha sweep covers 100 values from 1.889 to 1800. The zero-alpha case is
excluded: without tumor killing, these coefficients allow immune populations
to grow beyond float64 range over the corrected long observation window.

## Plots and interpretation

`Visualization.pop_plotting.plot_dixon` displays four nondimensional states,
three physical cell populations, physical IL-2, and the single treatment
input in separate, correctly labeled panels. Its horizontal axis is days.
Zero values are masked on logarithmic population axes, rather than replaced
by an arbitrary positive population.

The alpha analysis reports statistics and sampled peaks over the final 25%
of the physical window. These are finite-time summaries, not a demonstration
of a bifurcation. For positive alpha, substituting `Z=alpha*C2` removes alpha
from the system; alpha alone cannot change equilibrium stability.

`Analysis_Dixon.ipynb` runs the Dixon alpha-sweep peak diagrams, tumor feature
extraction, FFT in cycles/day, trajectory embedding, and descriptive K-means
clustering directly, without a legacy analysis flag. UMAP is used when
`umap-learn` is installed; otherwise the embedding is explicitly labeled PCA.
Alpha labels are excluded from the clustering features. The unsupported KP
GA comparisons have been removed from the Dixon analysis. The single-run Dixon GA
cells in `main_dixon.ipynb` use five genes ordered `[C1, C2, I, T, 1]` to
control one nonnegative IL-2 input `s`. Fitness prediction and integration
share the same Dixon coefficients. Run the GA parameter, setup, and simulation
cells in order; output is `Output_data/simu_data_dixon_ga1.csv`, including
physical states and the IL-2 source rate `s_bar = s*K*lambda_tumor`. The later Dixon GA alpha sweep uses the same alpha values as the
no-treatment sweep and writes one CSV per run under
`Output_data/dixon_alpha_sweep_ga_v2/`. It resets the populations and GA for each
alpha, retains log cell integration, and saves Dixon coefficients and physical
scales with the results. Resume skips complete files with matching saved
settings; a mismatch requires a new output directory or
`overwrite_existing=True`.

The Dixon fitness predictor now has an optional compiled C implementation of
exactly the same stability-limited RK4 calculation. Build once from the project
root with `python3 -m GA.native_fitness_dixon --build`, then restart the kernel.
The default `fitness_backend='auto'` uses C when available, otherwise Python;
set the environment's backend to `'python'` or `'native'` to choose explicitly.
The notebook prints the selected backend and shows time-step progress within
each alpha run. No generations, population sizes, or observation points were
reduced for this acceleration.

The GA uses continuous gene bounds `{'low': -10.0, 'high': 10.0}`. The previous
`(-10.0, 10.0)` tuple restricted each gene to only the two endpoints in PyGAD.
The corrected sweep uses a new output directory and records a controller
version so prior results are not silently reused. The default 203 alpha values,
1,999 control intervals, 30 generations, and 100 candidates still constitute a
large computation; compiled prediction does not make the full sweep instant.
