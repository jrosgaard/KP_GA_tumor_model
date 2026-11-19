/*
 * Population fitness for the non-dimensional Kirschner-Panetta model.
 * Compare this with GeneticAlgorithmKP.fitness_func in fitness_function_KP.py.
 *
 * Python stores the arrays. C borrows their addresses for this call,
 * reads the inputs, and writes one result per candidate into fitness_out.
 * There is no malloc/free here: no memory is allocated or retained by C.
 */
#include <stddef.h>  /* size_t: an unsigned integer type for sizes and indices. */

/*
 * void means that this function does not return a value. Its output is an array.
 * double is a floating-point number matching NumPy's float64 on this build.
 * A pointer (the *) holds the address of an element in memory. const prevents
 * this function from changing the input elements through those pointers.
 *
 * Contract (checked by the Python wrapper):
 *   population: count rows of 8 genes, contiguous in row-major order.
 *   state:      [x, y, z].
 *   params:     [c, mu_2, p_1, g_1, r_2, b, alpha, g_2, p_2, g_3, mu_3].
 *   weights:    [a1, a2, b1, b2, b3, b4, c1, c2].
 *   fitness_out: space for count doubles; must not overlap the inputs.
 * The caller must provide valid pointers and nonzero model denominators.
 */
void population_fitness(
    const double *population,
    size_t count,
    const double *state,
    const double *params,
    const double *weights,
    double t_step,
    double *fitness_out
)
{
    /* C arrays start at index 0. A declaration also specifies the value's type.
     * const local variables make values that should stay fixed easy to spot. */
    const double x = state[0];
    const double y = state[1];
    const double z = state[2];
    const double c = params[0];
    const double mu_2 = params[1];
    const double p_1 = params[2];
    const double g_1 = params[3];
    const double r_2 = params[4];
    const double b = params[5];
    const double alpha = params[6];
    const double g_2 = params[7];
    const double p_2 = params[8];
    const double g_3 = params[9];
    const double mu_3 = params[10];

    const double a1 = weights[0];
    const double a2 = weights[1];
    const double b1 = weights[2];
    const double b2 = weights[3];
    const double b3 = weights[4];
    const double b4 = weights[5];
    const double c1 = weights[6];
    const double c2 = weights[7];

    /* These derivatives do not depend on the candidate's treatment inputs.
     * Compute them once per batch, rather than once per candidate. */
    const double dx_base = c*y - mu_2*x + (p_1*x*z)/(g_1 + z);
    const double dy = r_2*y*(1.0 - b*y) - (alpha*x*y)/(g_2 + y);
    const double dz_base = (p_2*x*y)/(g_3 + y) - mu_3*z;
    const double y_pred = y + dy*t_step;

    /* for(initialization; condition; update). ++i increases i by one.
     * Braces enclose the statements executed for each candidate. */
    for (size_t i = 0; i < count; ++i) {
        /* Pointer arithmetic is in elements, not bytes. Adding 8*i skips
         * i rows, each containing eight doubles. genes[0] is that row's
         * first gene; genes[7] is its last. This does not copy the row. */
        const double *genes = population + 8*i;
        double s_1 = genes[0]*x + genes[1]*y + genes[2]*z + genes[3];
        double s_2 = genes[4]*x + genes[5]*y + genes[6]*z + genes[7];

        /* if works like Python's if, with parentheses and braces.
         * Negative treatment is clipped exactly as in solution_to_inputs. */
        if (s_1 < 0.0) {
            s_1 = 0.0;
        }
        if (s_2 < 0.0) {
            s_2 = 0.0;
        }

        /* The reference fitness uses one forward-Euler prediction.
         * This is NOT the RK45 integration used to advance the simulation. */
        const double x_pred = x + (dx_base + s_1)*t_step;
        const double z_pred = z + (dz_base + s_2)*t_step;
        const double immunotherapy = a1*x_pred - a2*y_pred;
        /* C's ^ is bitwise XOR, not exponentiation. Square with z_pred*z_pred. */
        const double toxicity = b1*s_2 + b2*s_1 + b3*(x_pred*s_2)
                              + b4*(z_pred*z_pred);

        /* Store into Python's output array. A semicolon ends each statement. */
        fitness_out[i] = c1*immunotherapy - c2*toxicity;
    }
}
