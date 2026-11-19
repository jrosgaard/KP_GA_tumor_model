"""One-input Dixon fitness, compared with an independent adaptive prediction."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from GA.fitness_function_Dixon import GeneticAlgorithmDixon as GA
from Model.Dixon_model import dixon_coupled


@pytest.fixture
def environment():
    return dict(t=0, C1=0.1, C2=0.002, I=0.003, T=0.4, t_step=0.36)


def test_five_genes_control_only_s_in_state_order():
    state = [.2, .3, .4, .5]
    for index, expected in enumerate([*state, 1]):
        genes = np.eye(5)[index]
        assert GA.solution_to_inputs(genes, *state) == expected
    assert GA.solution_to_inputs([-1]*5, *state) == 0
    with pytest.raises(ValueError, match='5 genes'):
        GA.solution_to_inputs([1]*8, *state)


@pytest.mark.parametrize('dose', [0, .01, .1])
def test_fitness_matches_adaptive_dixon_prediction(environment, dose):
    ga = SimpleNamespace(environment=environment)
    params = GA.DEFAULT_MODEL_PARAMS
    weights = GA.DEFAULT_FITNESS_WEIGHTS
    initial = [environment[k] for k in ['C1', 'C2', 'I', 'T']]
    arguments = [dose] + [params[k] for k in
        ['beta', 'mu_0', 'k1', 'mu_1', 'k2', 'mu_2', 'sigma', 'g', 'alpha']]
    prediction = solve_ivp(dixon_coupled, [0, environment['t_step']], initial,
        args=tuple(arguments), method='DOP853', rtol=1e-12, atol=1e-14).y[:, -1]
    _, c2, il, tumor = prediction
    expected = (weights['c1']*(weights['a1']*c2 - weights['a2']*tumor)
                - weights['c2']*(weights['b1']*dose + weights['b3']*c2*dose + weights['b4']*il**2))
    assert GA.fitness_func(ga, [0, 0, 0, 0, dose], 0) == pytest.approx(expected, rel=2e-6, abs=1e-8)


def test_dose_reaches_cd8_and_tumor_in_fitness_prediction(environment):
    # Remove toxicity to isolate the downstream benefit of IL-2 treatment.
    ga = SimpleNamespace(environment=dict(environment,
        fitness_weights=dict(a1=1, a2=1, c1=1, c2=0)))
    untreated = GA.fitness_func(ga, [0]*5, 0)
    treated = GA.fitness_func(ga, [0, 0, 0, 0, .1], 0)
    assert treated > untreated


def test_batch_matches_scalar_and_does_not_mutate_inputs(environment):
    population = np.random.default_rng(5300).uniform(-.1, .1, (10, 5))
    saved = population.copy()
    ga = SimpleNamespace(environment=environment)
    batch = GA.fitness_func_batch(ga, population, None)
    scalar = [GA.fitness_func(ga, row, i) for i, row in enumerate(population)]
    np.testing.assert_allclose(batch, scalar, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(population, saved)
    assert GA.fitness_func_batch(ga, np.empty((0, 5)), None).shape == (0,)


def test_fixed_s_in_environment_does_not_override_the_genes(environment):
    population = [[0, 0, 0, 0, .05]]
    baseline = GA.fitness_func_batch(SimpleNamespace(environment=environment), population, None)
    other = dict(environment, s=100, model_params=dict(s=200))
    np.testing.assert_array_equal(
        GA.fitness_func_batch(SimpleNamespace(environment=other), population, None), baseline)


def test_top_level_model_override_has_precedence(environment):
    ga = SimpleNamespace(environment=dict(environment, alpha=0,
        model_params=dict(alpha=715.72)))
    result = GA.fitness_func(ga, [0]*5, 0)
    ga.environment = dict(environment, model_params=dict(alpha=0))
    assert GA.fitness_func(ga, [0]*5, 0) == result


@pytest.mark.parametrize('changes', [
    dict(C1=-1), dict(I=np.nan), dict(t_step=-1), dict(t_step=np.inf),
    dict(prediction_step=0), dict(model_params=dict(g=0)),
])
def test_invalid_environment(environment, changes):
    with pytest.raises(ValueError):
        GA.fitness_func(SimpleNamespace(environment=dict(environment, **changes)), [0]*5, 0)


def test_actual_pygad_uses_five_gene_single_input(environment):
    pygad = pytest.importorskip('pygad')
    ga = pygad.GA(num_generations=2, num_parents_mating=2, sol_per_pop=6,
        num_genes=5, fitness_func=GA.fitness_func_batch, fitness_batch_size=3,
        init_range_low=0, init_range_high=.1, mutation_num_genes=1,
        on_generation=GA.on_generation, random_seed=5300)
    ga.environment = environment
    ga.run()
    solution, fitness, _ = ga.best_solution(pop_fitness=ga.last_generation_fitness)
    assert solution.shape == (5,)
    assert np.isfinite(fitness)
    assert GA.solution_to_inputs(solution, *[environment[k] for k in ['C1', 'C2', 'I', 'T']]) >= 0
