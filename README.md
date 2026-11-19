# Kirschner-Panetta and Dixon Tumor immunotherapy model

This project aimed to use a genetic algorithm to perform closed-loop optimization of immunotherapy dosing in the Kirschner-Panetta model and then the Dixon model. The genetic algorithm was allowed to control the effector cell and IL-2 dosing at each step in the Kirschner-Panetta model, while in the Dixon model only the IL-2 treatment term was controlled.

Using the slider notebook, you are able to adjust the antigenicity (c) for both the model without treatment and with the GA-optimized treatment regime.

## Project structure

- `GA/`: Genetic algorithm and fitness function logic.
- `Model/`: KP and Dixon models with integration helper.
- `Visualization/`: Plotting and data handling functions.
- `Output_data/`: Generated simulation CSVs of antigenicity sweeps.
- `Figures/`: Generated figures and exports.
- `docs/`: Literature notes and project presentation.
- `*.ipynb`: Notebooks kept at the root so their existing relative paths continue to work.

For the corrected Dixon clock, published parameter comparison, four-state
plots, and output locations, see [Dixon simulation](docs/dixon_simulation.md).

## GA module names

Use `from GA.fitness_function_KP import GeneticAlgorithmKP` for KP and
`from GA.fitness_function_Dixon import GeneticAlgorithmDixon` for Dixon.
The Dixon controller uses five genes to control a single nonnegative IL-2
input and predicts all four Dixon states with stability-limited RK4.

Each wrapper builds a separate shared library. Build the optional Dixon
accelerator with `python3 -m GA.native_fitness_dixon --build` (lowercase `dixon`),
then restart the kernel. Dixon uses this library automatically when available
and falls back to Python otherwise.

## C population fitness evaluator

The GA configurations in `main_kp.ipynb` evaluate population fitness in C. Build
the library once before running them (a C compiler is required):

```sh
python3 -m GA.native_fitness_KP --build
```

Read the annotated source in `GA/population_fitness_KP.c` and the
[C walkthrough](docs/c_fitness_walkthrough.md) for pointers, array layout,
Python integration, and instructions for testing or using the Python reference.

## C integrators

Build the annotated Euler and RK4 integrators with:

```sh
python3 -m Model.native_integration --build
```

Select them using `integrator.integrate(state, params, t_span, method="rk4",
t_step=0.001)` (or `method="euler"`). Existing calls keep using adaptive RK45.
See the [integration walkthrough](docs/c_integration_walkthrough.md) for the
C structure, parameter order, and how to choose and check the internal step.

The C solver supports KP only. `dixon_integrate` uses SciPy RK45 by default,
also supports SciPy Radau for the published parameter set, and provides
Python implementations for `method="euler"` and `method="rk4"`, with the
same `t_step` option. Its non-dimensional state order is `[C1, C2, I, T]`
and its parameter order is `[s, beta, mu_0, k1, mu_1, k2, mu_2, sigma, g, alpha]`.



