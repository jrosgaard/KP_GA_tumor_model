/* Fixed-step integrators for the non-dimensional Kirschner-Panetta model.
 *
 * Read in this order: kp_rhs -> euler_step -> rk4_step -> integrate.
 * All functions are at file scope: standard C does not allow nested functions.
 * Python owns the input/output buffers; C uses small local arrays for scratch
 * space. There is no malloc/free, and no pointer is retained after a call.
 */
#include <math.h>    /* isfinite checks for NaN and infinity. */
#include <stddef.h>  /* size_t is an unsigned type for sizes/indices. */
#include <string.h>  /* strcmp compares the contents of C strings. */

/* Named integer constants make the return codes easier to read.
 * Returning an error lets Python raise an exception without terminating Python.
 */
enum {
    KP_OK = 0,
    KP_INVALID_INPUT = 1,
    KP_UNKNOWN_METHOD = 2,
    KP_NUMERICAL_ERROR = 3,
    KP_TIME_STALLED = 4
};

/* static makes this helper private to this source file.
 * const double * means we may read the doubles through this pointer, but may
 * not change them. derivative is writable and receives [dx/dt, dy/dt, dz/dt].
 *
 * params has the SAME 13-value order as Model/integration.py:
 * [c, mu_2, p_1, g_1, s_1, r_2, b, alpha, g_2, p_2, g_3, mu_3, s_2].
 * This differs from the fitness evaluator's 11-value parameter array!
 */
static int kp_rhs(const double *state, const double *params, double *derivative)
{
    const double x = state[0];
    const double y = state[1];
    const double z = state[2];
    const double c = params[0];
    const double mu_2 = params[1];
    const double p_1 = params[2];
    const double g_1 = params[3];
    const double s_1 = params[4];
    const double r_2 = params[5];
    const double b = params[6];
    const double alpha = params[7];
    const double g_2 = params[8];
    const double p_2 = params[9];
    const double g_3 = params[10];
    const double mu_3 = params[11];
    const double s_2 = params[12];
    const double denominators[3] = {g_1 + z, g_2 + y, g_3 + y};

    /* Check at EVERY stage: RK4 also evaluates intermediate states. */
    for (size_t j = 0; j < 3; ++j) {
        if (!isfinite(state[j]) || !isfinite(denominators[j])
            || denominators[j] == 0.0) {
            return KP_NUMERICAL_ERROR;
        }
    }

    derivative[0] = c*y - mu_2*x + (p_1*x*z)/(g_1 + z) + s_1;
    derivative[1] = r_2*y*(1.0 - b*y) - (alpha*x*y)/(g_2 + y);
    derivative[2] = (p_2*x*y)/(g_3 + y) - mu_3*z + s_2;
    for (size_t j = 0; j < 3; ++j) {
        if (!isfinite(derivative[j])) {
            return KP_NUMERICAL_ERROR;
        }
    }
    return KP_OK;
}

/* Euler: u_next = u + h*f(u), where u is the vector [x, y, z]. */
static int euler_step(double *state, const double *params, double h)
{
    double derivative[3];
    const int status = kp_rhs(state, params, derivative);
    if (status != KP_OK) {
        return status;
    }
    /* All derivatives were calculated BEFORE changing any state variable. */
    for (size_t j = 0; j < 3; ++j) {
        state[j] += h*derivative[j];
    }
    return KP_OK;
}

/* Classical fourth-order Runge-Kutta: four slope estimates per step.
 * Each k array stores a derivative, not a derivative multiplied by h.
 * This is fixed-step RK4, not the adaptive RK45 used by SciPy.
 */
static int rk4_step(double *state, const double *params, double h)
{
    double k1[3], k2[3], k3[3], k4[3], temporary[3];
    int status = kp_rhs(state, params, k1);
    if (status != KP_OK) {
        return status;
    }

    /* First midpoint estimate: use the slope at the start. */
    for (size_t j = 0; j < 3; ++j) {
        temporary[j] = state[j] + (h/2.0)*k1[j];
    }
    status = kp_rhs(temporary, params, k2);
    if (status != KP_OK) {
        return status;
    }

    /* Second midpoint estimate. Start from the ORIGINAL state again. */
    for (size_t j = 0; j < 3; ++j) {
        temporary[j] = state[j] + (h/2.0)*k2[j];
    }
    status = kp_rhs(temporary, params, k3);
    if (status != KP_OK) {
        return status;
    }

    /* Endpoint estimate, using the second midpoint slope. */
    for (size_t j = 0; j < 3; ++j) {
        temporary[j] = state[j] + h*k3[j];
    }
    status = kp_rhs(temporary, params, k4);
    if (status != KP_OK) {
        return status;
    }

    /* Combine all four slopes only after all stages are complete. */
    for (size_t j = 0; j < 3; ++j) {
        state[j] += (h/6.0)*(k1[j] + 2.0*k2[j] + 2.0*k3[j] + k4[j]);
    }
    return KP_OK;
}

/* Public entry point, called from Python through ctypes.
 *
 * Buffer lengths: state[3], params[13], t_span[2], state_out[3]. The caller
 * must supply valid buffers of these sizes; C pointers do not carry a length.
 * method must be a null-terminated string: "euler" or "rk4".
 * Forward integration only; t_step is a positive INTERNAL step size.
 * Treatment inputs and parameters stay constant over this call's interval.
 * On success, write the final state into state_out and return KP_OK (zero).
 * On failure, return an error code and leave state_out unchanged.
 */
int integrate(const double *state, const double *params, const double *t_span,
              double t_step, const char *method, double *state_out)
{
    if (!state || !params || !t_span || !method || !state_out) {
        return KP_INVALID_INPUT;
    }
    if (!isfinite(t_step) || t_step <= 0.0
        || !isfinite(t_span[0]) || !isfinite(t_span[1])
        || t_span[1] < t_span[0] || !isfinite(t_span[1] - t_span[0])) {
        return KP_INVALID_INPUT;
    }
    for (size_t j = 0; j < 3; ++j) {
        if (!isfinite(state[j])) {
            return KP_INVALID_INPUT;
        }
    }
    for (size_t j = 0; j < 13; ++j) {
        if (!isfinite(params[j])) {
            return KP_INVALID_INPUT;
        }
    }

    /* strcmp returns ZERO when strings match; == compares pointer addresses
     * if used directly on strings. Check the full name, not just its first letter.
     */
    const int use_euler = strcmp(method, "euler") == 0;
    if (!use_euler && strcmp(method, "rk4") != 0) {
        return KP_UNKNOWN_METHOD;
    }

    /* Copy the initial state: the caller's input must remain unchanged. */
    double current[3] = {state[0], state[1], state[2]};
    double t = t_span[0];
    const double t_end = t_span[1];
    while (t < t_end) {
        const double remaining = t_end - t;
        /* condition ? a : b selects a if true, otherwise b. Shortening the last
         * step stops at t_end even when the interval is not divisible by t_step.
         */
        const double h = t_step < remaining ? t_step : remaining;
        if (t + h == t) {
            return KP_TIME_STALLED;
        }

        const int status = use_euler ? euler_step(current, params, h)
                                     : rk4_step(current, params, h);
        if (status != KP_OK) {
            return status;
        }
        for (size_t j = 0; j < 3; ++j) {
            if (!isfinite(current[j])) {
                return KP_NUMERICAL_ERROR;
            }
        }
        t = h == remaining ? t_end : t + h;
    }

    /* A zero-length interval skips the loop and returns the initial state. */
    for (size_t j = 0; j < 3; ++j) {
        state_out[j] = current[j];
    }
    return KP_OK;
}
