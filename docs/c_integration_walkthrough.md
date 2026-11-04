# Learning C through the KP integrators

Build the shared library from the project root:

```sh
python3 -m Model.native_integration --build
```

The build supports macOS and Linux and requires a C compiler (`cc`). Rebuild
after editing `Model/integration.c`, then restart your notebook kernel so it
loads the new machine code.

The existing Python entry point accepts two optional C methods:

```python
from Model.integration import kp_integrate

# Same order as the original Python integration function, including both doses.
params = [0.02, 0.03, 0.1245, 2e4, 0.0,
          0.18, 1e-5, 0.002, 1e5, 5e-7, 1e4, 10.0, 0.0]
state = [1.0, 1.0, 1.0]
integrator = kp_integrate()

euler = integrator.integrate(state, params, (0, 0.36), method="euler", t_step=0.001)
rk4 = integrator.integrate(state, params, (0, 0.36), method="rk4", t_step=0.001)
reference = integrator.integrate(state, params, (0, 0.36))  # Existing SciPy RK45.
```

For an existing simulation loop, add `method="rk4", t_step=...` to its
`integrator.integrate(...)` call to select C. Existing calls still use RK45.
`t_step` is the internal numerical step, in the same non-dimensional time units
as `t_span`. It can be much smaller than the interval between controller updates.
Parameters and dosing remain constant during each call, matching the existing
Python integrator. The function returns the final `[x, y, z]`, not a trajectory.

## How the C file is organized

Read the functions in this order:

1. `kp_rhs` computes all three derivatives from one state. It has no time loop.
2. `euler_step` advances a state once using `u_new = u + h*f(u)`.
3. `rk4_step` uses four slope estimates to advance a state once.
4. `integrate` validates inputs, copies the initial state, and repeats steps
   until it reaches the end of the interval. It shortens the last step as needed.

C function definitions sit at file scope, outside other functions. A helper's
`static` keyword makes it private to this source file. This is why the step
functions go above `integrate`, rather than inside its `if` branches.

`const double *state` borrows read-only input data. `double current[3]` is a
mutable local copy. `double *state_out` points to Python's output buffer. These
are distinct roles: the derivatives helper writes derivatives, and the public
integrator writes the final state only after successful integration.

`enum` gives names to integer error codes. Zero means success here. Python
converts nonzero return codes into exceptions. This allows a numerical error
to be reported without calling `exit` and closing the notebook process.

`strcmp(method, "euler") == 0` tests the full string contents for equality.
The `==` operator on two string pointers would compare addresses instead.
Methods are exactly `"euler"` and `"rk4"`; leapfrog is not implemented.

## Tracing RK4

The arrays `k1` through `k4` hold slopes (derivatives), not increments:

```text
k1 = f(current)
k2 = f(current + h*k1/2)
k3 = f(current + h*k2/2)
k4 = f(current + h*k3)
next = current + h*(k1 + 2*k2 + 2*k3 + k4)/6
```

Every temporary state starts from the original `current`. Only the last loop
updates `current`. All three components of a temporary state must be ready
before calling `kp_rhs` because the equations are coupled.

## Accuracy and choosing a step

These C methods use fixed steps. They do not implement RK45's adaptive error
control or its `rtol` and `atol`. The default C step of 0.01 is a starting value,
not an accuracy guarantee. Neither method guarantees positive populations.

Compare against RK45 for your parameters, then halve the C step and compare
again. For smooth solutions with sufficiently small stable steps, Euler's
global error is first order and RK4's is fourth order. The tests check both
convergence rates using equations with known analytic solutions.

The IL-2 loss term can make step selection particularly important. For the
isolated decay equation `z' = -mu_3*z`, Euler's amplification factor is
`1 - h*mu_3`; it decays in magnitude only when `0 < h*mu_3 < 2` for positive
`mu_3`. This is a diagnostic for that scalar term, not a stability guarantee
for the complete coupled model.

The C code checks zero denominators and nonfinite arithmetic at intermediate
stages, and detects when a step cannot advance floating-point time. Finite
but inaccurate results still require step refinement checks.

Run the tests with:

```sh
python3 -m pytest tests/test_native_integration.py
```
