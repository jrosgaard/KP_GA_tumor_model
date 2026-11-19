"""Accuracy, convergence, and error handling for the annotated C integrators."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from Model.KP_model import kp_coupled
from Model.integration import kp_integrate
from Model.native_integration import build_library, integrate_native, _load_library


@pytest.fixture(scope="session", autouse=True)
def compiled_integrator():
    build_library()


@pytest.fixture
def params():
    # Simplified model with analytic solutions: x/z decay with constant dosing,
    # while y grows exponentially. All model denominators remain nonzero.
    return np.array([0, 0.7, 0, 1, 0.2, 0.3, 0, 0, 1, 0, 1, 1.2, 0.4], dtype=float)


def exact_solution(t):
    return np.array([
        0.2/0.7 + (1 - 0.2/0.7)*np.exp(-0.7*t),
        2*np.exp(0.3*t),
        0.4/1.2 + (3 - 0.4/1.2)*np.exp(-1.2*t),
    ])


@pytest.mark.parametrize("method,minimum_ratio", [("euler", 1.8), ("rk4", 14)])
def test_halving_step_improves_accuracy_at_expected_order(params, method, minimum_ratio):
    errors = []
    for h in (0.1, 0.05):
        result = integrate_native([1, 2, 3], params, [0, 1], method=method, t_step=h)
        errors.append(np.linalg.norm(result - exact_solution(1)))
    assert errors[0]/errors[1] > minimum_ratio


@pytest.mark.parametrize("method,tolerance", [("euler", 3e-4), ("rk4", 1e-9)])
def test_shortened_last_step_and_treatment_inputs(params, method, tolerance):
    # Neither 0.37 nor its translation is divisible by h=0.001.
    result = integrate_native([1, 2, 3], params, [2, 2.3704], method=method, t_step=0.001)
    np.testing.assert_allclose(result, exact_solution(0.3704), rtol=tolerance)


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_coupled_model_agrees_with_scipy(method):
    params = [0.165, 1/6, 0.1245/0.18, 2e5, 0.2,
              1, 1e-4, 1/0.18, 1, 5e5/18, 0.01, 10/0.18, 0.4]
    state = [1., 1., 1.]
    reference = solve_ivp(kp_coupled, [0, 0.36], state, args=tuple(params),
                          rtol=1e-12, atol=1e-13).y[:, -1]
    # Euler needs a smaller step to meet this accuracy target than RK4 does.
    h = 0.00025 if method == "euler" else 0.0005
    result = integrate_native(state, params, [0, 0.36], method=method, t_step=h)
    np.testing.assert_allclose(result, reference, rtol=2e-4 if method == "euler" else 1e-8)


def test_zero_interval_and_input_ownership(params):
    storage = np.array([1., 99, 2, 99, 3, 99])
    state = storage[::2]  # A noncontiguous input must also work.
    before = storage.copy()
    np.testing.assert_array_equal(integrate_native(state, params, [2, 2]), state)
    integrate_native(state, params, [0, 0.1])
    np.testing.assert_array_equal(storage, before)


@pytest.mark.parametrize("kwargs", [
    {"state": [1, 2]}, {"params": [1]*11}, {"t_span": [0]},
    {"state": [1, np.nan, 2]}, {"t_span": [0, np.inf]},
    {"t_span": [1, 0]}, {"t_step": 0}, {"t_step": -1},
    {"t_step": np.nan}, {"method": "leapfrog"}, {"method": "r"},
])
def test_invalid_inputs(params, kwargs):
    arguments = dict(state=[1, 2, 3], params=params, t_span=[0, 1])
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        integrate_native(**arguments)


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_zero_denominator(params, method):
    with pytest.raises(FloatingPointError, match="denominator"):
        integrate_native([1, -1, 3], params, [0, 0.1], method=method)


def test_step_cannot_advance_time(params):
    with pytest.raises(FloatingPointError, match="too small"):
        integrate_native([1, 2, 3], params, [1e16, 1e16 + 2], t_step=0.01)


def test_c_failure_does_not_write_output(params):
    state = np.array([1., -1, 3])
    output = np.full(3, -999.)
    status = _load_library().integrate(state, params, np.array([0., 0.1]),
                                      0.01, b"rk4", output)
    assert status == 3
    np.testing.assert_array_equal(output, [-999]*3)


def test_python_entry_point_preserves_default_and_supports_c(params):
    integrator = kp_integrate()
    default = integrator.integrate([1, 2, 3], params, [0, 0.37])
    np.testing.assert_allclose(default, exact_solution(0.37), rtol=1e-7)
    for method in ("euler", "rk4"):
        result = integrator.integrate([1, 2, 3], params, [0, 0.37], method=method, t_step=0.001)
        assert isinstance(result, list)
        np.testing.assert_allclose(result, exact_solution(0.37), rtol=3e-4)
    with pytest.raises(ValueError, match="method"):
        integrator.integrate([1, 2, 3], params, [0, 1], method="invalid")


@pytest.mark.parametrize('method,tolerance', [('euler', 3e-3), ('rk4', 2e-6)])
@pytest.mark.parametrize('effector', [100., 1000.])
def test_fast_tumor_killing_stays_nonnegative_and_matches_stiff_solver(method, tolerance, effector):
    # The former h=.01 solver produced y=-221 at x=100 in this interval.
    # These are the notebook's nondimensional KP coefficients.
    params = [0.165, 1/6, 0.1245/0.18, 2e5, 0,
              1, 1e-4, 1/0.18, 1, 5e5/18, .01, 10/0.18, 0]
    state = [effector, .001, 1.]
    reference = solve_ivp(kp_coupled, [0, .36], state, args=tuple(params),
                          method='Radau', rtol=1e-11, atol=1e-14)
    assert reference.success
    result = integrate_native(state, params, [0, .36], method=method)
    assert np.isfinite(result).all() and (result >= 0).all()
    # Euler still needs refinement for accuracy even after stability is fixed.
    if method == 'euler':
        result = integrate_native(state, params, [0, .36], method=method, t_step=1e-6)
    np.testing.assert_allclose(result, reference.y[:, -1], rtol=tolerance, atol=1e-9)


@pytest.mark.parametrize('method', ['euler', 'rk4'])
def test_large_dose_never_accepts_negative_intermediate_tumor(method):
    params = [0.165, 1/6, 0.1245/0.18, 2e5, 1e5,
              1, 1e-4, 1/0.18, 1, 5e5/18, .01, 10/0.18, 0]
    result = integrate_native([0, .001, 1], params, [0, .01], method=method)
    assert np.isfinite(result).all() and (result >= 0).all()
    reference = solve_ivp(kp_coupled, [0, .01], [0, .001, 1], args=tuple(params),
                          method='Radau', rtol=1e-11, atol=1e-14)
    # Stability does not supply error control; verify convergence with a
    # suitably small maximum step for this deliberately abrupt treatment.
    result = integrate_native([0, .001, 1], params, [0, .01], method=method,
                              t_step=1e-6 if method == 'euler' else 1e-4)
    np.testing.assert_allclose(result, reference.y[:, -1],
                               rtol=.04 if method == 'euler' else 1e-4, atol=1e-9)


def test_loaded_library_reports_stability_limited_version():
    from Model.native_integration import integrator_version
    assert integrator_version() == 3


def test_stale_library_is_rejected_before_any_simulation(monkeypatch):
    from types import SimpleNamespace
    from Model import native_integration as native
    version = lambda: 1
    native._load_library.cache_clear()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(native.ctypes, 'CDLL', lambda path: SimpleNamespace(kp_integrator_version=version))
            with pytest.raises(RuntimeError, match='incompatible'):
                native.integrator_version()
    finally:
        native._load_library.cache_clear()


def test_signed_antigenicity_experiment_matches_independent_solver():
    params = [-.005/.18, 1/6, .1245/.18, 2e5, 0,
              1, 1e-4, 1/.18, 1, 5e5/18, .01, 10/.18, 0]
    state = [-200., 1e4, -1e5]
    reference = solve_ivp(kp_coupled, [0, .36], state, args=tuple(params),
                          method='Radau', rtol=1e-11, atol=1e-12)
    assert reference.success
    actual = integrate_native(state, params, [0, .36], method='rk4')
    assert actual[0] < 0 and actual[1] > 0 and actual[2] < 0
    np.testing.assert_allclose(actual, reference.y[:, -1], rtol=2e-6, atol=1e-9)


def test_signed_fast_tumor_killing_cannot_numerically_flip_tumor():
    # A negative c previously disabled every stability check, even while x>0.
    params = [-.005/.18, 1/6, .1245/.18, 2e5, 0,
              1, 1e-4, 1/.18, 1, 5e5/18, .01, 10/.18, 0]
    state = [1914., .001, 1.]
    reference = solve_ivp(kp_coupled, [0, .36], state, args=tuple(params),
                          method='Radau', rtol=1e-11, atol=1e-14)
    actual = integrate_native(state, params, [0, .36], method='rk4')
    assert actual[1] >= 0
    np.testing.assert_allclose(actual, reference.y[:, -1], rtol=2e-6, atol=1e-9)


@pytest.mark.parametrize('method', ['euler', 'rk4'])
def test_signed_trial_cannot_jump_across_a_model_pole(method):
    params = [0, 0, .01, 1, 0, 0, 0, 0, 1, 0, 1, 0, -1]
    with pytest.raises(FloatingPointError):
        integrate_native([1, 1, -.5], params, [0, 1], method=method, t_step=1)
