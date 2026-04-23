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



