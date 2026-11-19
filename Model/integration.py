# Integration module
# This module integrates the KP model equations dxdt, dydt, and dzdt using solve_ivp Runge-Kutta method.
# The integration method could be changed later to include other models.

# Imports
import scipy
from scipy.integrate import solve_ivp
import numpy as np

# Import equations from KP model
from Model.KP_model import dx_dt, dy_dt, dz_dt, kp_coupled

# Import equations from Dixon model
from Model.Dixon_model import dixon_coupled, dixon_coupled_log, jacobian_log_dixon

# Integration class for KP model
class kp_integrate:
    def __init__(self):
        pass

    def integrate(self, state, params, t_span, *, method="RK45", t_step=0.01):
        """
        Runge-Kutta integrator.
        Variables:
        time step
        state [x, y, z]
        params [c, mu_2, p_1, g_1, s_1, r_2, b, alpha, g_2, p_2, g_3, mu_3, s_2]
        method: "RK45" (existing adaptive solver), "euler", or "rk4" (C).
        t_step: maximum internal step for C methods; reduced for stability
                with nonnegative states/coefficients; ignored by RK45.
        """

        if method in ("euler", "rk4"):
            from Model.native_integration import integrate_native

            return integrate_native(
                state, params, t_span, method=method, t_step=t_step
            ).tolist()
        if method != "RK45":
            raise ValueError("method must be 'RK45', 'euler', or 'rk4'.")

        # Integrate the coupled KP model equations
        solution = solve_ivp(fun=kp_coupled, t_span=t_span, y0=state, method="RK45", # uses Runge-Kutta method (RK45)
                             args=tuple(params), rtol=1e-7, atol=1e-9, max_step=0.1)

        # Get the final values of x, y, z
        x_out, y_out, z_out = solution.y[:,-1]

        # output
        return [x_out, y_out, z_out]


# Integration class for Dixon model (beta version)
class dixon_integrate:
    def __init__(self):
        pass

    def integrate(self, state, params, t_span, *, method="RK45", t_step=0.01,
                  log_cells=False, rtol=1e-8, atol=1e-11):
        """
        Runge-Kutta integrator.
        Variables:
        time step
        state [C1, C2, I, T]
        params [s, beta, mu_0, k1, mu_1, k2, mu_2, sigma, g, alpha]
        method: "RK45", "Radau", "DOP853", "BDF", or "LSODA" (SciPy);
        "euler" or "rk4" (Python).
        t_step: internal fixed step; ignored by RK45.
        Uses the non-dimensional Dixon equations (tumor carrying capacity 1).
        log_cells=True: input/output use [log(C1), log(C2), I, log(T)].
        Carry this log state between steps to preserve underflowed populations.
        Use to_log_state/from_log_state to convert; -inf encodes exact zeros.
        """
        if method not in ("RK45", "Radau", "DOP853", "BDF", "LSODA", "euler", "rk4"):
            raise ValueError("Unknown Dixon integration method.")

        buffers = []
        for name, value, shape in (
            ("state", state, (4,)), ("params", params, (10,)), ("t_span", t_span, (2,))
        ):
            array = np.asarray(value, dtype=float)
            valid = np.isfinite(array)
            if name == "state" and log_cells and array.shape == shape:
                valid[[0, 1, 3]] |= np.isneginf(array[[0, 1, 3]])
            if array.shape != shape or not valid.all():
                raise ValueError(f"{name} must be finite and have shape {shape}.")
            buffers.append(array)
        state, params, t_span = buffers
        if t_span[1] < t_span[0] or not np.isfinite(t_span[1] - t_span[0]):
            raise ValueError("Dixon integration requires a finite forward interval.")
        rhs_function = dixon_coupled
        options = {}
        active = None
        if log_cells:
            if (params < 0).any() or params[8] <= 0:
                raise ValueError("Log-state parameters must be nonnegative, with g > 0.")
            active = np.isfinite(state[[0, 1, 3]])
            state = state.copy()
            state[[0, 1, 3]] = np.where(active, state[[0, 1, 3]], 0)
            rhs_function = lambda t, u, *p: dixon_coupled_log(t, u, *p, active=active)
            if method in ("Radau", "BDF", "LSODA"):
                options['jac'] = lambda t, u, *p: jacobian_log_dixon(t, u, *p, active=active)

        def finish(current):
            current = np.asarray(current).copy()
            if log_cells:
                if active[2] and current[3] > 1e-6:
                    raise FloatingPointError("Dixon tumor left T <= 1; tighten solver tolerances.")
                current[[0, 1, 3]] = np.where(active, current[[0, 1, 3]], -np.inf)
            return current.tolist()

        if method in ("euler", "rk4"):
            t_step = float(t_step)
            if not np.isfinite(t_step) or t_step <= 0:
                raise ValueError("t_step must be finite and positive.")
            def rhs(t, current):
                with np.errstate(divide="raise", invalid="raise", over="raise"):
                    derivative = np.asarray(rhs_function(t, current, *params))
                if not np.isfinite(derivative).all():
                    raise FloatingPointError("Dixon integration produced nonfinite derivatives.")
                return derivative

            current = state.copy()
            t, t_end = t_span
            while t < t_end:
                remaining = t_end - t
                h = min(t_step, remaining)
                if t + h == t:
                    raise FloatingPointError("t_step is too small to advance time at this t_span.")
                with np.errstate(divide="raise", invalid="raise", over="raise"):
                    k1 = rhs(t, current)
                    if method == "euler":
                        current += h * k1
                    else:
                        k2 = rhs(t + h/2, current + h*k1/2)
                        k3 = rhs(t + h/2, current + h*k2/2)
                        k4 = rhs(t + h, current + h*k3)
                        current += h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t = t_end if h == remaining else t + h
            return finish(current)

        # Integrate the coupled Dixon model equations
        solution = solve_ivp(fun=rhs_function, t_span=t_span, y0=state,
                             method=method, args=tuple(params), rtol=rtol,
                             atol=atol, max_step=0.5, **options)

        if not solution.success:
            raise RuntimeError(f"Dixon integration failed: {solution.message}")

        # Get the final values of C1, C2, I, T
        C1_out, C2_out, I_out, T_out = solution.y[:,-1]

        # output
        return finish([C1_out, C2_out, I_out, T_out])


    @staticmethod
    def to_log_state(state):
        """Convert [C1, C2, I, T] to log cell coordinates; preserve exact zeros."""
        result = np.asarray(state, dtype=float).copy()
        if result.shape != (4,) or not np.isfinite(result).all() or (result < 0).any():
            raise ValueError("state must contain four finite, nonnegative values.")
        if result[3] > 1:
            raise ValueError("Initial tumor fraction T must be at most 1.")
        with np.errstate(divide="ignore"):
            result[[0, 1, 3]] = np.log(result[[0, 1, 3]])
        return result

    @staticmethod
    def from_log_state(state):
        """Convert a saved log state back to [C1, C2, I, T] for plotting."""
        result = np.asarray(state, dtype=float).copy()
        result[[0, 1, 3]] = np.exp(result[[0, 1, 3]])
        if result[2] < -1e-10:
            raise FloatingPointError("Dixon IL-2 became negative; tighten solver tolerances.")
        result[2] = max(0.0, result[2])
        return result
