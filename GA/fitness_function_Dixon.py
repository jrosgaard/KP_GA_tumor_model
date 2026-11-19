# Genetic Algorithm module
# One nonnegative IL-2 input s controlled by a five-gene affine controller.
# State order: non-dimensional [C1, C2, I, T].

import numpy as np

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Model.Dixon_model import dixon_dI_dt, dixon_dC1_dt, dixon_dC2_dt, dixon_dT_dt


class GeneticAlgorithmDixon:
    """
    GA callbacks for the non-dimensional Dixon model.

    Use num_genes=5. Genes encode coefficients of [C1, C2, I, T, 1]
    for the single treatment s. Other model parameters stay fixed during
    candidate evaluation. environment['t_step'] is in nondimensional time.

    Fitness = c1*(a1*C2_pred - a2*T_pred)
              - c2*(b1*s + b3*C2_pred*s + b4*I_pred**2).
    Weights retain the KP naming convention; b2 (a second dose) is removed.
    """

    DEFAULT_MODEL_PARAMS = {
        "K": 1e9,
        "beta": 15.67*1e-8,
        "mu_0": 55.56,
        "k1": 3.43,
        "mu_1": 0.93,
        "k2": 1.67,
        "mu_2": 0.93,
        "sigma": 0.692,
        "g": 0.02,
        "alpha": 715.72,
    }

    DEFAULT_FITNESS_WEIGHTS = {
        "a1": 0.1,
        "a2": 0.1,
        "b1": 0.1,
        "b3": 0.1,
        "b4": 0.1,
        "c1": 1.0,
        "c2": 3.0,
    }

    def __init__(self, parameters=None, id=None, ga_instance=None):
        self.parameters = parameters or {}
        self.id = id
        self.ga_instance = ga_instance
        self.last_fitness = None

    @staticmethod
    def _get_environment(ga_instance):
        environment = getattr(ga_instance, "environment", None)
        if environment is None:
            raise AttributeError(
                "GA instance is missing an 'environment' mapping with the current state."
            )

        missing = [key for key in ("t", "C1", "C2", "T", "I") if key not in environment]
        if missing:
            raise KeyError(
                f"GA environment is missing required keys: {', '.join(missing)}"
            )

        return environment

    @classmethod
    def _get_model_params(cls, environment):
        model_params = dict(cls.DEFAULT_MODEL_PARAMS)
        model_params.update(environment.get("model_params", {}))

        for key in cls.DEFAULT_MODEL_PARAMS:
            if key in environment:
                model_params[key] = environment[key]

        return model_params

    @classmethod
    def _get_fitness_weights(cls, environment):
        fitness_weights = dict(cls.DEFAULT_FITNESS_WEIGHTS)
        fitness_weights.update(environment.get("fitness_weights", {}))
        return fitness_weights

    @staticmethod
    def solution_to_inputs(solution, C1, C2, I, T):
        """Return the single input s from five genes ordered [C1, C2, I, T, 1]."""
        genes = np.asarray(solution, dtype=float)
        state = np.asarray([C1, C2, I, T], dtype=float)
        if genes.shape != (5,):
            raise ValueError("Expected exactly 5 genes for the single Dixon input s.")
        if not np.isfinite(genes).all() or not np.isfinite(state).all():
            raise ValueError("Genes and Dixon state must be finite.")
        return max(0.0, float(genes[:4] @ state + genes[4]))

    @staticmethod
    def _predict_states(state, params, treatment, horizon, prediction_step):
        """Vectorized RK4 prediction with s held constant over the horizon.

        Several stages are necessary: s changes I, then C2, then T. A single
        Euler prediction makes C2/T independent of s and rewards zero dosing
        whenever dosing is penalized. Shorten steps for fast decay/killing.
        This predicts fitness only; the notebook's integrator advances the
        actual simulation.
        """
        current = np.repeat(state[None, :], len(treatment), axis=0)
        if not len(current):
            return current

        def rhs(values):
            C1, C2, I, T = values.T
            return np.column_stack((
                dixon_dC1_dt(params['k1'], C1, T, params['mu_1']),
                dixon_dC2_dt(params['k2'], C2, T, params['mu_2'], params['sigma'], I, params['g']),
                dixon_dI_dt(params['beta'], C1, I, params['mu_0'], treatment),
                dixon_dT_dt(T, params['alpha'], C2),
            ))

        elapsed = 0.0
        with np.errstate(over='raise', divide='raise', invalid='raise'):
            while elapsed < horizon:
                maximum_rate = max(
                    1.0, params['mu_0'], params['mu_1'], params['mu_2'],
                    params['k1'] * current[:, 3].max(),
                    params['k2'] * current[:, 3].max() + params['sigma'],
                    params['alpha'] * current[:, 1].max(),
                    1.0 + current[:, 3].max(),
                )
                remaining = horizon - elapsed
                h = min(remaining, prediction_step, 0.5 / maximum_rate)
                if elapsed + h == elapsed:
                    raise FloatingPointError("Dixon fitness prediction cannot advance time.")
                k1 = rhs(current)
                k2 = rhs(current + h*k1/2)
                k3 = rhs(current + h*k2/2)
                k4 = rhs(current + h*k3)
                current += h/6 * (k1 + 2*k2 + 2*k3 + k4)
                if not np.isfinite(current).all() or (current < 0).any():
                    raise FloatingPointError("Dixon fitness prediction became nonfinite or negative.")
                elapsed = horizon if h == remaining else elapsed + h
        return current

    @staticmethod
    def _evaluate_population(ga_instance, solutions):
        environment = GeneticAlgorithmDixon._get_environment(ga_instance)
        params = GeneticAlgorithmDixon._get_model_params(environment)
        weights = GeneticAlgorithmDixon._get_fitness_weights(environment)
        population = np.asarray(solutions, dtype=float)
        if population.ndim != 2 or population.shape[1] != 5:
            raise ValueError("Population must have shape (number_of_candidates, 5).")
        state = np.asarray([environment[key] for key in ('C1', 'C2', 'I', 'T')], dtype=float)
        horizon = float(environment.get('t_step', 1.0))
        prediction_step = float(environment.get('prediction_step', 0.01))
        coefficients = np.asarray([params[key] for key in
            ('beta', 'mu_0', 'k1', 'mu_1', 'k2', 'mu_2', 'sigma', 'g', 'alpha')], dtype=float)
        weight_values = np.asarray([weights[key] for key in ('a1', 'a2', 'b1', 'b3', 'b4', 'c1', 'c2')])
        if not all(np.isfinite(value).all() for value in (population, state, coefficients, weight_values)):
            raise ValueError("Population, Dixon state, coefficients, and weights must be finite.")
        if (state < 0).any() or state[3] > 1:
            raise ValueError("Dixon state must be nonnegative with T <= 1.")
        if (coefficients < 0).any() or params['g'] <= 0:
            raise ValueError("Dixon coefficients must be nonnegative with g > 0.")
        if not np.isfinite(horizon) or horizon < 0:
            raise ValueError("t_step must be finite and nonnegative.")
        if not np.isfinite(prediction_step) or prediction_step <= 0:
            raise ValueError("prediction_step must be finite and positive.")

        from GA.native_fitness_dixon import native_available, predict_states
        backend = environment.get('fitness_backend', 'auto')
        if backend not in ('auto', 'python', 'native'):
            raise ValueError("fitness_backend must be 'auto', 'python', or 'native'.")
        predictor = GeneticAlgorithmDixon._predict_states
        if backend == 'native' or (backend == 'auto' and native_available()):
            predictor = predict_states

        with np.errstate(over='raise', invalid='raise'):
            treatment = np.maximum(0.0, population[:, :4] @ state + population[:, 4])
            prediction = predictor(
                state, params, treatment, horizon, prediction_step,
            )
            C1_pred, C2_pred, I_pred, T_pred = prediction.T
            immunotherapy = weights['a1']*C2_pred - weights['a2']*T_pred
            toxicity = (weights['b1']*treatment + weights['b3']*C2_pred*treatment
                        + weights['b4']*I_pred**2)
            return weights['c1']*immunotherapy - weights['c2']*toxicity

    @staticmethod
    def fitness_func_batch(ga_instance, solutions, solution_indices):
        """PyGAD batch callback using the Dixon equations and one input s.

        Uses compiled RK4 prediction when built, otherwise vectorized Python.
        environment['fitness_backend'] can select 'auto', 'native', or 'python'.
        Set fitness_batch_size > 1 and num_genes=5 in pygad.GA.
        """
        del solution_indices
        return GeneticAlgorithmDixon._evaluate_population(ga_instance, solutions)

    @staticmethod
    def fitness_func(ga_instance, solution, solution_idx):
        """Scalar PyGAD callback, with the same fitness as the batch callback."""
        del solution_idx
        genes = np.asarray(solution, dtype=float)
        if genes.shape != (5,):
            raise ValueError("Expected exactly 5 genes for the single Dixon input s.")
        return float(GeneticAlgorithmDixon._evaluate_population(ga_instance, genes[None, :])[0])

    @staticmethod
    def on_start(ga_instance):
        del ga_instance
        print("on_start()")

    @staticmethod
    def on_fitness(ga_instance, population_fitness):
        del ga_instance, population_fitness
        print("on_fitness()")

    @staticmethod
    def on_parents(ga_instance, selected_parents):
        del ga_instance, selected_parents
        print("on_parents()")

    @staticmethod
    def on_crossover(ga_instance, offspring_crossover):
        del ga_instance, offspring_crossover
        print("on_crossover()")

    @staticmethod
    def on_mutation(ga_instance, offspring_mutation):
        del ga_instance, offspring_mutation
        print("on_mutation()")

    @staticmethod
    def on_generation(ga_instance):
        if not hasattr(ga_instance, "last_generation_fitness"):
            return

        _, current_fitness, _ = ga_instance.best_solution(
            pop_fitness=ga_instance.last_generation_fitness
        )
        ga_instance.last_fitness = current_fitness

    @staticmethod
    def on_stop(ga_instance, last_population_fitness):
        del ga_instance, last_population_fitness
        print("on_stop()")

    def attach(self, ga_instance):
        self.ga_instance = ga_instance
        return ga_instance

    def run(self, environment):
        if self.ga_instance is None:
            raise RuntimeError(
                "No PyGAD instance is attached. Pass one to the constructor or call attach()."
            )

        self.ga_instance.environment = environment
        self.ga_instance.run()

        best_solution, best_fitness, best_solution_idx = self.ga_instance.best_solution(
            pop_fitness=self.ga_instance.last_generation_fitness
        )
        self.last_fitness = best_fitness

        return best_solution, best_fitness, best_solution_idx
