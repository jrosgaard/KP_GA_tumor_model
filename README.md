# Kirschner Panetta Tumor immunotherapy model

This project aimed to use a genetic algorithm to perform closed-loop optimization of immunotherapy dosing in the Kirschner-Panetta model. 

Using the slider notebook, you are able to adjust the antigenicity (c) for both the model without treatment and with the GA-optimized treatment regime.

## Project structure

- `GA/`: genetic algorithm controller and fitness logic
- `Model/`: KP and Dixon model equations plus integration helpers
- `Visualization/`: plotting and data handling utilities
- `Output_data/`: generated simulation CSVs
- `Figures/`: reserved for generated figures and exports
- `docs/`: literature notes and presentation material
- `*.ipynb`: working notebooks kept at the repo root so their existing relative paths continue to work



