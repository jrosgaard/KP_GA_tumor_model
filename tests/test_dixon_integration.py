"""Check Dixon state/parameter ordering and all supported integration methods."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from Model.Dixon_model import dixon_coupled
from Model.integration import dixon_integrate


def test_derivative_order():
    # Distinct entries expose state or parameter permutations.
    result = dixon_coupled(0, [2., 3., 5., 0.4],
                          0.7, 0.8, 0.9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.25)
    np.testing.assert_allclose(result, [
        0.2*2*0.4 - 0.3*2,
        0.4*3*0.4 + 0.6*3*5/(0.1 + 5) - 0.5*3,
        0.8*2 - 0.9*5 + 0.7,
        0.4*(1 - 0.4) - 0.25*3*0.4,
    ])


@pytest.mark.parametrize("method,tolerance", [("RK45", 1e-7), ("euler", 5e-4), ("rk4", 1e-10)])
def test_coupled_solution(method, tolerance):
    state = [0.8, 0.6, 0.2, 0.4]
    params = [0.1, 0.7, 0.9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.25]

    def reference_rhs(t, u):
        c1, c2, il, tumor = u
        return [0.2*c1*tumor - 0.3*c1,
                0.4*c2*tumor + 0.6*c2*il/(0.1 + il) - 0.5*c2,
                0.7*c1 - 0.9*il + 0.1,
                tumor*(1 - tumor) - 0.25*c2*tumor]

    expected = solve_ivp(reference_rhs, [2, 2.3704], state,
                         rtol=1e-12, atol=1e-13).y[:, -1]
    result = dixon_integrate().integrate(state, params, [2, 2.3704],
                                          method=method, t_step=0.001)
    assert isinstance(result, list)
    np.testing.assert_allclose(result, expected, rtol=tolerance)


@pytest.mark.parametrize("method,minimum_ratio", [("euler", 1.8), ("rk4", 14)])
def test_convergence(method, minimum_ratio):
    # With no coupling and zero tumor, all three immune components decay.
    params = [0, 0, 0.7, 0, 0.3, 0, 0.5, 0, 1, 0]
    state = np.array([1., 2., 3., 0.])
    expected = state * np.exp(-np.array([0.3, 0.5, 0.7, 0]))
    errors = [np.linalg.norm(dixon_integrate().integrate(
        state, params, [0, 1], method=method, t_step=h) - expected)
        for h in (0.1, 0.05)]
    assert errors[0]/errors[1] > minimum_ratio
    np.testing.assert_array_equal(state, [1, 2, 3, 0])


@pytest.mark.parametrize("method", ["RK45", "euler", "rk4"])
def test_zero_interval(method):
    state = [1., 2., 3., 0.]
    assert dixon_integrate().integrate(state, [1]*10, [2, 2], method=method) == state


@pytest.mark.parametrize("kwargs", [
    {"state": [1, 2, 3]}, {"params": [1]*13}, {"t_span": [0]},
    {"state": [1, 2, np.nan, 4]}, {"t_step": 0}, {"t_step": np.inf},
    {"t_span": [1, 0]}, {"method": "invalid"},
])
def test_invalid_inputs(kwargs):
    arguments = dict(state=[1, 2, 3, 4], params=[1]*10, t_span=[0, 1], method="rk4")
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        dixon_integrate().integrate(**arguments)


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_singular_denominator(method):
    with pytest.raises(FloatingPointError):
        dixon_integrate().integrate([1, 2, -1, 0.4], [1]*10, [0, 1], method=method)


def test_time_stalled():
    with pytest.raises(FloatingPointError, match="too small"):
        dixon_integrate().integrate([1]*4, [1]*10, [1e16, 1e16 + 2], method="rk4")


@pytest.mark.parametrize("method", ["RK45", "Radau", "DOP853", "BDF", "LSODA", "rk4"])
def test_log_states_agree_with_original_equations(method):
    integrator = dixon_integrate()
    state = [0.8, 0.6, 0.2, 0.4]
    params = [0.1, 0.7, 0.9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.25]
    expected = solve_ivp(dixon_coupled, [0, 0.36], state, args=tuple(params),
                         method='DOP853', rtol=1e-12, atol=1e-14).y[:, -1]
    log_state = integrator.to_log_state(state)
    for start, end in zip([0, 0.12, 0.24], [0.12, 0.24, 0.36]):
        log_state = integrator.integrate(log_state, params, [start, end],
            method=method, log_cells=True, t_step=0.001)
    np.testing.assert_allclose(integrator.from_log_state(log_state), expected, rtol=2e-7)


def test_log_state_retains_recovery_across_steps_after_underflow():
    integrator = dixon_integrate()
    params = [0, 0, 1, 0, 1, 0, 1, 0, 1, 1000]
    state = integrator.to_log_state([0, 1, 0, 0.2])
    state = integrator.integrate(state, params, [0, 20], method='Radau', log_cells=True)
    assert integrator.from_log_state(state)[3] == 0
    assert state[3]/np.log(10) < -400
    state = integrator.integrate(state, params, [20, 1100], method='Radau', log_cells=True)
    restored = integrator.from_log_state(state)
    assert restored[3] > 0.99
    assert restored[0] == 0


@pytest.mark.parametrize('state', [[1, 1, -1, 0.5], [1, 1, 1, 2], [1, 2, 3]])
def test_log_conversion_rejects_invalid_physical_states(state):
    with pytest.raises(ValueError):
        dixon_integrate.to_log_state(state)


def test_exact_zero_populations_survive_log_conversion_and_integration():
    integrator = dixon_integrate()
    log_state = integrator.to_log_state([0, 0, 0, 1])
    result = integrator.integrate(log_state, [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [0, 720], method='Radau', log_cells=True)
    np.testing.assert_array_equal(integrator.from_log_state(result), [0, 0, 0, 1])


def test_dixon_plot_keeps_tumor_and_treatment_in_their_own_panels():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from Visualization.pop_plotting import plot_dixon

    import pandas as pd
    data = pd.DataFrame(dict(t=[0, 1, 2], tau=[0, .18, .36],
        C1=[.1, .2, .3], C2=[.001, .002, .003], I=[.01, .02, .03],
        T=[.2, .3, .4], C1_bar=[1e8, 2e8, 3e8], C2_bar=[1e6, 2e6, 3e6],
        I_bar=[1e7, 2e7, 3e7], T_bar=[2e8, 3e8, 4e8], s=[.1, .1, .1]))
    fig, axes = plot_dixon(data)
    assert len(axes[0, 0].lines) == 4
    np.testing.assert_array_equal(axes[0, 0].lines[-1].get_ydata(), data['T'])
    np.testing.assert_array_equal(axes[0, 1].lines[-1].get_ydata(), data['T_bar'])
    np.testing.assert_array_equal(axes[1, 0].lines[0].get_ydata(), data['I_bar'])
    np.testing.assert_array_equal(axes[1, 1].lines[0].get_ydata(), data['s'])
    for ax in axes.flat:
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), data['t'])
    plt.close(fig)

