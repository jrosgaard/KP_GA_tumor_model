/* Batched Dixon RK4 prediction. State order: C1, C2, I, T.
 * Parameters: beta, mu_0, k1, mu_1, k2, mu_2, sigma, g, alpha.
 * Python owns all buffers. Keep the same shared stability step as Python. */
#include <math.h>
#include <stddef.h>

static void rhs(const double *u, const double *p, double dose, double *out)
{
    out[0] = p[2]*u[0]*u[3] - p[3]*u[0];
    out[1] = p[4]*u[1]*u[3] + (p[6]*u[1]*u[2])/(p[7] + u[2]) - p[5]*u[1];
    out[2] = p[0]*u[0] - p[1]*u[2] + dose;
    out[3] = u[3]*(1.0 - u[3]) - p[8]*u[1]*u[3];
}

/* Returns 1 if time cannot advance, 2 for invalid predicted states. */
int dixon_predict(size_t count, double *current, const double *params,
                  const double *treatment, double horizon, double prediction_step)
{
    double elapsed = 0.0;
    if (!count) return 0;
    while (elapsed < horizon) {
        double max_c2 = 0.0, max_t = 0.0;
        for (size_t i = 0; i < count; ++i) {
            max_c2 = fmax(max_c2, current[4*i + 1]);
            max_t = fmax(max_t, current[4*i + 3]);
        }
        double rate = fmax(1.0, fmax(params[1], fmax(params[3], params[5])));
        rate = fmax(rate, params[2]*max_t);
        rate = fmax(rate, params[4]*max_t + params[6]);
        rate = fmax(rate, params[8]*max_c2);
        rate = fmax(rate, 1.0 + max_t);
        const double remaining = horizon - elapsed;
        const double h = fmin(remaining, fmin(prediction_step, 0.5/rate));
        if (elapsed + h == elapsed) return 1;
        for (size_t i = 0; i < count; ++i) {
            double *u = current + 4*i;
            double k1[4], k2[4], k3[4], k4[4], trial[4];
            rhs(u, params, treatment[i], k1);
            for (size_t j = 0; j < 4; ++j) trial[j] = u[j] + h*k1[j]/2;
            rhs(trial, params, treatment[i], k2);
            for (size_t j = 0; j < 4; ++j) trial[j] = u[j] + h*k2[j]/2;
            rhs(trial, params, treatment[i], k3);
            for (size_t j = 0; j < 4; ++j) trial[j] = u[j] + h*k3[j];
            rhs(trial, params, treatment[i], k4);
            for (size_t j = 0; j < 4; ++j) {
                u[j] += h/6 * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j]);
                if (!isfinite(u[j]) || u[j] < 0.0) return 2;
            }
        }
        elapsed = h == remaining ? horizon : elapsed + h;
    }
    return 0;
}
