"""Compiled Dixon prediction must agree with the Python RK4 reference."""
from types import SimpleNamespace

import numpy as np
import pytest

from GA.fitness_function_Dixon import GeneticAlgorithmDixon as GA
from GA.native_fitness_dixon import build_library, predict_states


@pytest.fixture(scope='module', autouse=True)
def compiled_predictor():
    build_library()


@pytest.mark.parametrize('state,alpha,horizon', [
    ([1e-4, 1e-4, 1e-7, 1e-4], 1.889, 720/1999),
    ([.1, .002, .003, .4], 715.72, .36),
    ([.01, .1, .2, .7], 1700, .36),
    ([0, 0, 0, 0], 1, 1),
    ([.1, .002, .003, .4], 0, 0),
])
def test_compiled_prediction_matches_python(state, alpha, horizon):
    state = np.asarray(state, dtype=float)
    params = dict(GA.DEFAULT_MODEL_PARAMS, alpha=alpha)
    doses = np.array([0, .001, .1, 1, 10])
    expected = GA._predict_states(state, params, doses, horizon, .01)
    actual = predict_states(state, params, doses, horizon, .01)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


def test_native_and_python_population_fitness_match():
    population = np.random.default_rng(5300).uniform(-10, 10, (100, 5))
    saved = population.copy()
    environment = dict(t=0, C1=.1, C2=.002, I=.003, T=.4, t_step=.36)
    outputs = [GA.fitness_func_batch(SimpleNamespace(environment=dict(environment,
               fitness_backend=backend)), population, None) for backend in ('native', 'python')]
    np.testing.assert_allclose(*outputs, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(population, saved)


def test_auto_falls_back_without_a_library(monkeypatch):
    import GA.native_fitness_dixon as native
    monkeypatch.setattr(native, 'native_available', lambda: False)
    def unexpected_native_call(*args):
        raise AssertionError('Native predictor should not be called')
    monkeypatch.setattr(native, 'predict_states', unexpected_native_call)
    environment = dict(t=0, C1=.1, C2=.002, I=.003, T=.4, t_step=.36)
    instance = SimpleNamespace(environment=environment)
    auto = GA.fitness_func(instance, [0, 0, 0, 0, .1], 0)
    instance.environment['fitness_backend'] = 'python'
    assert GA.fitness_func(instance, [0, 0, 0, 0, .1], 0) == auto


@pytest.mark.parametrize('changes', [dict(t_step=-1), dict(prediction_step=0),
    dict(C1=-1), dict(I=np.nan), dict(model_params=dict(g=0)), dict(fitness_backend='bad')])
def test_native_rejects_invalid_environment(changes):
    environment = dict(t=0, C1=.1, C2=.002, I=.003, T=.4, t_step=.36, fitness_backend='native')
    environment.update(changes)
    with pytest.raises(ValueError):
        GA.fitness_func(SimpleNamespace(environment=environment), [0]*5, 0)
