# Kirschner Panetta Tumor immunotherapy model

This project aimed to use a genetic algorithm to perform closed-loop optimization of immunotherapy dosing in the Kirschner-Panetta model.
The genetic algorithm was allowed to control the effector cell and IL-2 dosing at each step.

Using the slider notebook, you are able to adjust the antigenicity (c) for both the model without treatment and with the GA-optimized treatment regime.

## Project structure

- `GA/`: Genetic algorithm and fitness function logic.
- `Model/`: KP and Dixon models with integration helper using RK45.
- `Visualization/`: Plotting and data handling functions.
- `Output_data/`: Generated simulation CSVs of antigenicity sweeps.
- `Figures/`: Generated figures and exports.
- `docs/`: Literature notes and project presentation.
- `*.ipynb`: Notebooks kept at the root so their existing relative paths continue to work.

## C population fitness evaluator

The GA configurations in `main.ipynb` evaluate population fitness in C. Build
the library once before running them (a C compiler is required):

```sh
python3 -m GA.native_fitness --build
```

Read the annotated source in `GA/population_fitness.c` and the
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



