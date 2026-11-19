"""Compiled Dixon RK4 predictor. Build with python3 -m GA.native_fitness_dixon --build."""
import argparse
import ctypes
from functools import lru_cache
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np

MODEL_KEYS = ('beta', 'mu_0', 'k1', 'mu_1', 'k2', 'mu_2', 'sigma', 'g', 'alpha')
SOURCE_PATH = Path(__file__).with_name('population_fitness_Dixon.c')
LIBRARY_PATH = SOURCE_PATH.with_name(
    '_population_fitness_Dixon.dylib' if sys.platform == 'darwin' else '_population_fitness_Dixon.so')


def build_library():
    """Compile explicitly, once; never compile from the fitness callback."""
    if sys.platform not in ('darwin', 'linux'):
        raise RuntimeError('The build command supports macOS and Linux.')
    command = shlex.split(os.environ.get('CC', 'cc')) + [
        '-std=c99', '-O3', '-Wall', '-Wextra', '-Werror', '-ffp-contract=off',
        '-fPIC', '-dynamiclib' if sys.platform == 'darwin' else '-shared',
        str(SOURCE_PATH), '-o', str(LIBRARY_PATH), '-lm',
    ]
    subprocess.run(command, check=True)
    _load_library.cache_clear()
    native_available.cache_clear()
    return LIBRARY_PATH


@lru_cache(maxsize=1)
def _load_library():
    try:
        library = ctypes.CDLL(str(LIBRARY_PATH))
        function = library.dixon_predict
    except (OSError, AttributeError) as error:
        raise RuntimeError('Build Dixon fitness with python3 -m GA.native_fitness_dixon --build, '
                           'then restart your kernel.') from error
    vector = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    matrix = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS')
    function.argtypes = [ctypes.c_size_t, matrix, vector, vector, ctypes.c_double, ctypes.c_double]
    function.restype = ctypes.c_int
    return library


@lru_cache(maxsize=1)
def native_available():
    """Cache missing-library detection so Python fallback stays inexpensive."""
    try:
        _load_library()
    except RuntimeError:
        return False
    return True


def predict_states(state, model_params, treatment, horizon, prediction_step):
    """Predict the four Dixon states for each constant dose; never mutate inputs."""
    state = np.asarray(state, dtype=np.float64)
    params = np.array([model_params[key] for key in MODEL_KEYS], dtype=np.float64)
    doses = np.ascontiguousarray(treatment, dtype=np.float64)
    horizon, prediction_step = float(horizon), float(prediction_step)
    if state.shape != (4,) or doses.ndim != 1:
        raise ValueError('Expected four Dixon states and a one-dimensional treatment array.')
    if not all(np.isfinite(x).all() for x in (state, params, doses)):
        raise ValueError('Dixon states, parameters, and treatment must be finite.')
    if (state < 0).any() or state[3] > 1 or (params < 0).any() or params[7] <= 0 or (doses < 0).any():
        raise ValueError('Dixon inputs must be nonnegative, with T <= 1 and g > 0.')
    if not np.isfinite(horizon) or horizon < 0 or not np.isfinite(prediction_step) or prediction_step <= 0:
        raise ValueError('Require finite horizon >= 0 and prediction_step > 0.')
    current = np.repeat(state[None, :], len(doses), axis=0)
    if len(doses):
        status = _load_library().dixon_predict(len(doses), current, params, doses, horizon, prediction_step)
        if status:
            raise FloatingPointError('Dixon fitness prediction cannot advance time.' if status == 1
                                     else 'Dixon fitness prediction became nonfinite or negative.')
    return current


def evaluate_population(population, state, model_params, fitness_weights, t_step):
    """Evaluate five-gene controllers using the validated Dixon fitness interface."""
    from types import SimpleNamespace
    from GA.fitness_function_Dixon import GeneticAlgorithmDixon
    state = np.asarray(state, dtype=float)
    if state.shape != (4,):
        raise ValueError('State must contain exactly [C1, C2, I, T].')
    environment = dict(zip(('C1', 'C2', 'I', 'T'), state))
    environment.update(t=0, model_params=model_params, fitness_weights=fitness_weights,
                       t_step=t_step, fitness_backend='native')
    return GeneticAlgorithmDixon.fitness_func_batch(SimpleNamespace(environment=environment), population, None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true', help='Compile the C shared library.')
    if parser.parse_args().build:
        print(f'Built {build_library()}')
    else:
        parser.print_help()
