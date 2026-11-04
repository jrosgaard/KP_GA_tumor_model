# Integration module
# This module integrates the KP model equations dxdt, dydt, and dzdt using solve_ivp Runge-Kutta method.
# The integration method could be changed later to include other models.

# Imports
import scipy
from scipy.integrate import solve_ivp
import numpy as np

# Import equations from KP model
from Model.KP_model import dx_dt, dy_dt, dz_dt, kp_coupled

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
        t_step: internal fixed step for C methods; ignored by RK45.
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

