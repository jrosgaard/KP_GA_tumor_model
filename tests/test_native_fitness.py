"""Check the compiled evaluator against the original Python equations.

Run from the project root: python3 -m pytest tests/test_native_fitness.py
A C compiler is required; the session fixture builds the library once.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from GA.fitness_function import GeneticAlgorithm
from GA.native_fitness import build_library, evaluate_population


@pytest.fixture(scope="session", autouse=True)
def compiled_library():
    build_library()


def reference(population, environment):
    ga = SimpleNamespace(environment=environment)
    return np.array([
        GeneticAlgorithm.fitness_func(ga, row, index)
        for index, row in enumerate(population)
    ])


@pytest.mark.parametrize("count", [1, 7, 100])
@pytest.mark.parametrize("scale", [1.0, 1e3])
def test_matches_reference_with_overrides(count, scale):
    rng = np.random.default_rng(5300)
    population = rng.uniform(-10, 10, (count, 8))
    environment = {
        "t": 12.0, "x": 1.2*scale, "y": 2.3*scale, "z": 0.7*scale,
        "t_step": 0.360180090045,
        "model_params": {
            "c": 0.165, "mu_2": 1/6, "p_1": 0.1245/0.18, "g_1": 2e5,
            "r_2": 1.0, "b": 1e-4, "alpha": 1/0.18, "g_2": 1.0,
            "p_2": 5e5/18, "g_3": 0.01, "mu_3": 10/0.18,
        },
        # Top-level model overrides must take precedence over the mapping.
        "c": 0.12,
        "fitness_weights": {"a1": 0.23, "b3": 0.04, "c2": 1.7},
    }
    actual = GeneticAlgorithm.fitness_func_batch(
        SimpleNamespace(environment=environment), population, list(range(count))
    )
    np.testing.assert_allclose(actual, reference(population, environment), rtol=1e-12, atol=1e-10)


def test_default_parameters_and_clipped_inputs():
    population = np.array([
        [-1.0]*8,  # Both treatment inputs are clipped to zero.
        [0.0]*8,
        [1.0]*8,
        [-1, 0, 0, 0, 0, 0, 0, 2],  # Only s_1 is clipped.
        [0, 0, 0, 2, -1, 0, 0, 0],  # Only s_2 is clipped.
    ])
    environment = {"t": 0, "x": 1.0, "y": 1.0, "z": 1.0}
    actual = GeneticAlgorithm.fitness_func_batch(SimpleNamespace(environment=environment), population, None)
    np.testing.assert_allclose(actual, reference(population, environment), rtol=1e-12)
    assert actual[0] == actual[1]


def test_noncontiguous_population_is_copied_without_mutation():
    storage = np.arange(80, dtype=np.float32).reshape(5, 16)
    population = storage[:, ::2]
    original = storage.copy()
    environment = {"t": 0, "x": 1.5, "y": 2.0, "z": 0.5}
    actual = GeneticAlgorithm.fitness_func_batch(SimpleNamespace(environment=environment), population, None)
    # The C boundary deliberately promotes float32 genes to float64 arithmetic.
    np.testing.assert_allclose(actual, reference(population.astype(np.float64), environment), rtol=1e-12)
    np.testing.assert_array_equal(storage, original)
    assert actual.dtype == np.float64


def evaluate(population, state=(1.0, 1.0, 1.0), t_step=1.0):
    return evaluate_population(
        population, state, GeneticAlgorithm.DEFAULT_MODEL_PARAMS,
        GeneticAlgorithm.DEFAULT_FITNESS_WEIGHTS, t_step,
    )


def test_empty_population():
    assert evaluate(np.empty((0, 8))).shape == (0,)


@pytest.mark.parametrize("population", [np.zeros(8), np.zeros((2, 7)), np.zeros((2, 9))])
def test_rejects_wrong_gene_shape(population):
    with pytest.raises(ValueError, match="shape"):
        evaluate(population)


@pytest.mark.parametrize("state", [(1, 2), [[1, 2, 3]]])
def test_rejects_wrong_state_shape(state):
    with pytest.raises(ValueError, match="State"):
        evaluate(np.zeros((2, 8)), state)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_inputs(invalid):
    with pytest.raises(ValueError, match="finite"):
        evaluate(np.full((2, 8), invalid))
    with pytest.raises(ValueError, match="finite"):
        evaluate(np.zeros((2, 8)), t_step=invalid)


def test_rejects_zero_model_denominator():
    with pytest.raises(ValueError, match="denominators"):
        evaluate(np.zeros((2, 8)), state=(1, -1e5, 1))


def test_actual_pygad_batch_run_matches_scalar_run():
    pygad = pytest.importorskip("pygad")
    population = np.random.default_rng(5300).uniform(-2, 2, (12, 8))
    environment = {"t": 0, "x": 1.5, "y": 2.0, "z": 0.5, "t_step": 0.36}
    runs = []
    for callback, batch_size in (
        (GeneticAlgorithm.fitness_func, None),
        (GeneticAlgorithm.fitness_func_batch, 5),  # Includes a short final batch.
    ):
        ga = pygad.GA(
            num_generations=4, num_parents_mating=4,
            initial_population=population.copy(), fitness_func=callback,
            fitness_batch_size=batch_size, random_seed=5300, mutation_num_genes=2,
            on_generation=GeneticAlgorithm.on_generation,
        )
        ga.environment = environment.copy()
        ga.run()
        runs.append(ga)
    np.testing.assert_array_equal(runs[0].population, runs[1].population)
    np.testing.assert_allclose(
        runs[0].last_generation_fitness, runs[1].last_generation_fitness, rtol=1e-12
    )
