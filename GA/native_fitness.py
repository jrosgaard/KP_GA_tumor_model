"""Build and call the C population evaluator using Python's standard ctypes.

Build from the project root with: python3 -m GA.native_fitness --build
The compiler is needed only for that command, not for each fitness evaluation.
"""

import argparse
import ctypes
from functools import lru_cache
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np


# These orders are part of the C interface; see population_fitness.c.
MODEL_KEYS = (
    "c", "mu_2", "p_1", "g_1", "r_2", "b", "alpha", "g_2", "p_2", "g_3", "mu_3"
)
WEIGHT_KEYS = ("a1", "a2", "b1", "b2", "b3", "b4", "c1", "c2")
SOURCE_PATH = Path(__file__).with_name("population_fitness.c")
LIBRARY_PATH = SOURCE_PATH.with_name(
    "_population_fitness.dylib" if sys.platform == "darwin" else "_population_fitness.so"
)


def build_library():
    """Compile a shared library for this machine (macOS or Linux)."""
    if sys.platform not in ("darwin", "linux"):
        raise RuntimeError("The build command currently supports macOS and Linux.")
    compiler = shlex.split(os.environ.get("CC", "cc"))
    # -O3 optimizes the loop; C99 permits declarations inside a for statement.
    # Disable fused multiply-add to keep rounding close to the Python reference.
    # Deliberately avoid -ffast-math, which relaxes floating-point semantics.
    command = compiler + [
        "-std=c99", "-O3", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
        "-fPIC", "-dynamiclib" if sys.platform == "darwin" else "-shared",
        str(SOURCE_PATH), "-o", str(LIBRARY_PATH),
    ]
    subprocess.run(command, check=True)
    return LIBRARY_PATH


@lru_cache(maxsize=1)
def _load_library():
    """Load once per Python process; keep the library alive with the function."""
    try:
        library = ctypes.CDLL(str(LIBRARY_PATH))
    except OSError as error:
        raise RuntimeError(
            "Cannot load the C fitness library. From the project root, run "
            "'python3 -m GA.native_fitness --build', then restart your kernel."
        ) from error

    # ctypes needs the exact C signature. Without it, a wrong pointer or integer
    # size could corrupt memory. ndpointer also checks dtype and array layout.
    vector = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    matrix = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
    library.population_fitness.argtypes = [
        matrix, ctypes.c_size_t, vector, vector, vector, ctypes.c_double, vector
    ]
    library.population_fitness.restype = None  # C's void return type.
    return library


def evaluate_population(population, state, model_params, fitness_weights, t_step):
    """Return one fitness per row; inputs must be finite and have eight genes.

    Arrays are converted to contiguous float64 buffers before their addresses
    cross into C. Local variables keep these buffers alive throughout the call.
    """
    population = np.ascontiguousarray(population, dtype=np.float64)
    state = np.ascontiguousarray(state, dtype=np.float64)
    params = np.array([model_params[key] for key in MODEL_KEYS], dtype=np.float64)
    weights = np.array([fitness_weights[key] for key in WEIGHT_KEYS], dtype=np.float64)
    t_step = float(t_step)

    if population.ndim != 2 or population.shape[1] != 8:
        raise ValueError("Population must have shape (number_of_candidates, 8).")
    if state.shape != (3,):
        raise ValueError("State must contain exactly [x, y, z].")
    if not all(np.isfinite(array).all() for array in (population, state, params, weights)):
        raise ValueError("Population, state, parameters, and weights must be finite.")
    if not np.isfinite(t_step):
        raise ValueError("t_step must be finite.")
    x, y, z = state
    if any(value == 0.0 for value in (params[3] + z, params[7] + y, params[9] + y)):
        raise ValueError("Model denominators g_1 + z, g_2 + y, g_3 + y must be nonzero.")

    # Python allocates the output; C fills every element before we return it.
    fitness = np.empty(population.shape[0], dtype=np.float64)
    if fitness.size:
        _load_library().population_fitness(
            population, population.shape[0], state, params, weights, t_step, fitness
        )
    return fitness


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Compile the C shared library.")
    arguments = parser.parse_args()
    if arguments.build:
        print(f"Built {build_library()}")
    else:
        parser.print_help()
