# Genetic Algorithm module
# This version made to work with non-dimensional variables x, y, z

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Model.KP_model import dx_dt, dy_dt, dz_dt


class GeneticAlgorithm:
    """
    Helper methods for the GA controller used with the non-dimensional KP model.
    """

    DEFAULT_MODEL_PARAMS = {
        "c": 0.02,
        "mu_2": 0.03,
        "p_1": 0.1245,
        "g_1": 2e4,
        "r_2": 0.18,
        "b": 1e-5,
        "alpha": 0.002,
        "g_2": 1e5,
        "p_2": 5e-7,
        "g_3": 1e4,
        "mu_3": 10,
    }

    DEFAULT_FITNESS_WEIGHTS = {
        "a1": 0.1,
        "a2": 0.1,
        "b1": 0.1,
        "b2": 0.1,
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

        missing = [key for key in ("t", "x", "y", "z") if key not in environment]
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
    def solution_to_inputs(solution, x, y, z):
        """
        Convert the 8-gene affine controller into the two treatment inputs.
        """
        if len(solution) < 8:
            raise ValueError(
                "Expected an 8-gene solution: 4 genes for s_1 and 4 genes for s_2."
            )

        genes1 = solution[0:4]
        genes2 = solution[4:8]

        s_1 = genes1[0] * x + genes1[1] * y + genes1[2] * z + genes1[3]
        s_2 = genes2[0] * x + genes2[1] * y + genes2[2] * z + genes2[3]

        return max(0, s_1), max(0, s_2)

    @staticmethod
    def fitness_func(ga_instance, solution, solution_idx):
        """
        Fitness function maximizing effector response while penalizing tumor burden
        and toxic dosing.
        """
        del solution_idx

        environment = GeneticAlgorithm._get_environment(ga_instance)
        model_params = GeneticAlgorithm._get_model_params(environment)
        fitness_weights = GeneticAlgorithm._get_fitness_weights(environment)

        t = environment["t"]
        x = environment["x"]
        y = environment["y"]
        z = environment["z"]
        t_step = environment.get("t_step", 1.0)

        s_1, s_2 = GeneticAlgorithm.solution_to_inputs(solution, x, y, z)

        x_pred = x + dx_dt(
            t=t,
            x=x,
            y=y,
            z=z,
            c=model_params["c"],
            mu_2=model_params["mu_2"],
            p_1=model_params["p_1"],
            g_1=model_params["g_1"],
            s_1=s_1,
        ) * t_step

        y_pred = y + dy_dt(
            t=t,
            y=y,
            x=x,
            z=z,
            r_2=model_params["r_2"],
            b=model_params["b"],
            alpha=model_params["alpha"],
            g_2=model_params["g_2"],
        ) * t_step

        z_pred = z + dz_dt(
            t=t,
            z=z,
            x=x,
            y=y,
            p_2=model_params["p_2"],
            g_3=model_params["g_3"],
            mu_3=model_params["mu_3"],
            s_2=s_2,
        ) * t_step

        immunotherapy = (
            fitness_weights["a1"] * x_pred - fitness_weights["a2"] * y_pred
        )
        toxicity = (
            fitness_weights["b1"] * s_2
            + fitness_weights["b2"] * s_1
            + fitness_weights["b3"] * (x_pred * s_2)
            + fitness_weights["b4"] * (z_pred**2)
        )

        return (
            fitness_weights["c1"] * immunotherapy
            - fitness_weights["c2"] * toxicity
        )

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
