/* Stability-limited integrators for the non-dimensional Kirschner-Panetta model.
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

/* Python verifies this symbol before calling the shared library. A changed
 * Python error message alone does not prove a notebook loaded the rebuilt C. */
int kp_integrator_version(void)
{
    return 3;
}

/* A numerical step must stay on the same side of every model pole. The tumor
 * equation is y'=y*f(...), so y cannot change sign in a regular solution.
 * Signed x and z remain permitted for negative-antigenicity experiments. */
static int same_domain(const double *trial, const double *reference, const double *params)
{
    if ((reference[1] >= 0.0 && trial[1] < 0.0)
        || (reference[1] <= 0.0 && trial[1] > 0.0)) return 0;
    const double old_den[3] = {params[3] + reference[2], params[8] + reference[1],
                               params[10] + reference[1]};
    const double new_den[3] = {params[3] + trial[2], params[8] + trial[1],
                               params[10] + trial[1]};
    for (size_t j = 0; j < 3; ++j) {
        if (!isfinite(new_den[j]) || old_den[j] == 0.0 || new_den[j] == 0.0
            || signbit(old_den[j]) != signbit(new_den[j])) return 0;
    }
    return 1;
}

/* static makes this helper private to this source file.
 * const double * means we may read the doubles through this pointer, but may
 * not change them. derivative is writable and receives [dx/dt, dy/dt, dz/dt].
 *
 * params has the SAME 13-value order as Model/integration.py:
 * [c, mu_2, p_1, g_1, s_1, r_2, b, alpha, g_2, p_2, g_3, mu_3, s_2].
 * This differs from the fitness evaluator's 11-value parameter array!
 */
static int kp_rhs(const double *state, const double *params, double *derivative,
                  int nonnegative, const double *reference)
{
    if (!same_domain(state, reference, params)) return KP_NUMERICAL_ERROR;
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
        if (!isfinite(state[j]) || (nonnegative && state[j] < 0.0)
            || !isfinite(denominators[j])
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
static int euler_step(double *state, const double *params, double h, int nonnegative)
{
    double derivative[3];
    const int status = kp_rhs(state, params, derivative, nonnegative, state);
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
 * Step sizes are limited for stability, without adaptive error control.
 */
static int rk4_step(double *state, const double *params, double h, int nonnegative)
{
    double k1[3], k2[3], k3[3], k4[3], temporary[3];
    int status = kp_rhs(state, params, k1, nonnegative, state);
    if (status != KP_OK) {
        return status;
    }

    /* First midpoint estimate: use the slope at the start. */
    for (size_t j = 0; j < 3; ++j) {
        temporary[j] = state[j] + (h/2.0)*k1[j];
    }
    status = kp_rhs(temporary, params, k2, nonnegative, state);
    if (status != KP_OK) {
        return status;
    }

    /* Second midpoint estimate. Start from the ORIGINAL state again. */
    for (size_t j = 0; j < 3; ++j) {
        temporary[j] = state[j] + (h/2.0)*k2[j];
    }
    status = kp_rhs(temporary, params, k3, nonnegative, state);
    if (status != KP_OK) {
        return status;
    }

    /* Endpoint estimate, using the second midpoint slope. */
    for (size_t j = 0; j < 3; ++j) {
        temporary[j] = state[j] + h*k3[j];
    }
    status = kp_rhs(temporary, params, k4, nonnegative, state);
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
 * Forward integration only; t_step is a positive maximum INTERNAL step size.
 * For nonnegative states and coefficients, limit it by the decay/killing rates
 * and reject trial stages that leave the nonnegative domain. This controls
 * stability, not truncation error: step convergence still needs checking.
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
    int nonnegative = 1;
    for (size_t j = 0; j < 3; ++j) {
        if (state[j] < 0.0) nonnegative = 0;
    }
    for (size_t j = 0; j < 13; ++j) {
        if (params[j] < 0.0) nonnegative = 0;
    }
    double t = t_span[0];
    const double t_end = t_span[1];
    while (t < t_end) {
        const double remaining = t_end - t;
        /* condition ? a : b selects a if true, otherwise b. Shortening the last
         * step stops at t_end even when the interval is not divisible by t_step.
         */
        double h = t_step < remaining ? t_step : remaining;
        {
            /* Bound both growth and decay rates, including signed experiments.
             * The stage checks below also prevent jumping across model poles. */
            const double il_den = params[3] + current[2];
            const double tumor_den = params[8] + current[1];
            if (il_den == 0.0 || tumor_den == 0.0) return KP_NUMERICAL_ERROR;
            double rate = fmax(fabs(params[1]), fabs(params[11]));
            rate = fmax(rate, fabs(params[2]*current[2]/il_den));
            rate = fmax(rate, fabs(params[5]*params[6]*current[1])
                              + fabs(params[7]*current[0]/tumor_den));
            if (!isfinite(rate)) return KP_NUMERICAL_ERROR;
            if (rate > 0.0) h = fmin(h, 0.5/rate);
        }
        if (t + h == t) {
            return KP_TIME_STALLED;
        }

        double trial[3];
        int accepted = 0;
        for (int attempt = 0; attempt < 50; ++attempt) {
            for (size_t j = 0; j < 3; ++j) trial[j] = current[j];
            const int status = use_euler ? euler_step(trial, params, h, nonnegative)
                                        : rk4_step(trial, params, h, nonnegative);
            int valid = status == KP_OK && same_domain(trial, current, params);
            for (size_t j = 0; j < 3; ++j) {
                if (!isfinite(trial[j]) || (nonnegative && trial[j] < 0.0)) valid = 0;
            }
            if (valid) { accepted = 1; break; }
            h /= 2.0;
            if (t + h == t) return KP_NUMERICAL_ERROR;
        }
        if (!accepted) return KP_NUMERICAL_ERROR;
        for (size_t j = 0; j < 3; ++j) current[j] = trial[j];
        t = h == remaining ? t_end : t + h;
    }

    /* A zero-length interval skips the loop and returns the initial state. */
    for (size_t j = 0; j < 3; ++j) {
        state_out[j] = current[j];
    }
    return KP_OK;
}
