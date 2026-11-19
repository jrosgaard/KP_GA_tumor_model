"""
ctypes bridge for integration.c. Build: python3 -m Model.native_integration --build
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

# Paths to the C source and compiled library.
SOURCE_PATH = Path(__file__).with_name("integration.c")
LIBRARY_PATH = SOURCE_PATH.with_name(
    "_integration_v3.dylib" if sys.platform == "darwin" else "_integration_v3.so"
)


def build_library():
    """
    Build machine-specific code; do this again after changing the C source.
    """
    if sys.platform not in ("darwin", "linux"):
        raise RuntimeError("The build command currently supports macOS and Linux.")
    command = shlex.split(os.environ.get("CC", "cc")) + [
        "-std=c99", "-O3", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
        "-fPIC", "-dynamiclib" if sys.platform == "darwin" else "-shared",
        str(SOURCE_PATH), "-o", str(LIBRARY_PATH), "-lm",
    ]
    subprocess.run(command, check=True)
    return LIBRARY_PATH


@lru_cache(maxsize=1)
def _load_library():
    """
    Load the compiled C library for integration.
    """
    try:
        library = ctypes.CDLL(str(LIBRARY_PATH))
        version = library.kp_integrator_version
        version.argtypes = []
        version.restype = ctypes.c_int
        if version() != 3:
            raise RuntimeError("Loaded an incompatible KP C integrator; rebuild and restart the kernel.")
    except (OSError, AttributeError) as error:
        raise RuntimeError(
            "Cannot load the C integrator. Run 'python3 -m Model.native_integration "
            "--build' from the project root, then restart your kernel."
        ) from error
    # C receives addresses of contiguous float64 arrays and a byte string.
    vector = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    library.integrate.argtypes = [
        vector, vector, vector, ctypes.c_double, ctypes.c_char_p, vector
    ]
    library.integrate.restype = ctypes.c_int  # Integer success/error code.
    return library


def integrator_version():
    """Return the version of the C code actually loaded in this process."""
    return _load_library().kp_integrator_version()


def integrate_native(state, params, t_span, *, method="rk4", t_step=0.01):
    """
    Return the final state from stability-limited forward integration.

    params must use the Python integrator's 13-value order, including s_1/s_2.
    t_step is the maximum internal step; nonnegative models may use smaller
    steps for stability and to keep nonnegative models nonnegative. Signed
    experiments retain signed x/z, but cannot jump across poles or flip y.
    Parameters are held constant. This is not adaptive error control.
    """
    if method not in ("euler", "rk4"):
        raise ValueError("C method must be 'euler' or 'rk4'.")
    buffers = []
    for name, value, shape in (
        ("state", state, (3,)), ("params", params, (13,)), ("t_span", t_span, (2,))
    ):
        array = np.ascontiguousarray(value, dtype=np.float64)
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}.")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite.")
        buffers.append(array)
    state, params, t_span = buffers
    t_step = float(t_step)
    if not np.isfinite(t_step) or t_step <= 0:
        raise ValueError("t_step must be finite and positive.")
    if t_span[1] < t_span[0]:
        raise ValueError("C integrators require t_end >= t_start.")

    # C writes into this output buffer only on success. Python keeps all arrays
    # alive for the duration of the call; C does not own or free these buffers.
    output = np.empty(3, dtype=np.float64)
    status = _load_library().integrate(
        state, params, t_span, t_step, method.encode("ascii"), output
    )
    if status in (1, 2):
        raise ValueError(f"C integrator rejected its inputs (status {status}).")
    if status == 3:
        raise FloatingPointError(
            "C integration encountered a zero model denominator, nonfinite arithmetic, "
            "or could not find a stable step within the current model domain. "
            f"method={method}, t_span={t_span.tolist()}, state={state.tolist()}."
        )
    if status == 4:
        raise FloatingPointError("t_step is too small to advance time at this t_span.")
    if status != 0:
        raise RuntimeError(f"Unknown C integrator status {status}.")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Compile the C integrators.")
    arguments = parser.parse_args()
    if arguments.build:
        print(f"Built {build_library()}")
    else:
        parser.print_help()
