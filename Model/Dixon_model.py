# Dixon model (non-dimensionalized)
# This is another immunotherapy model involving IL-2, CD4+, CD8+ and NK cells.
# This optional model uses the same notebook/integration structure as KP.

import numpy as np

# Dixon eqs. 1-4
def dixon_dI_dt(beta, C1, I, mu_0, s):
    """
    Equation for concentration of IL-2.
    beta: production rate of IL-2 by CD4+ cells
    C1: concentration of CD4+ cells
    I: concentration of IL-2
    mu_0: decay rate of IL-2
    s: treatment term or external source of IL-2
    """
    dI_dt = beta*C1 - mu_0*I + s
    return dI_dt

def dixon_dC1_dt(k1, C1, T, mu_1):
    """
    Equation for CD4+ cells.
    k1: stimulation rate of CD4+ cells by tumor cells
    C1: concentration of CD4+ cells
    T: concentration of tumor cells
    mu_1: death rate of CD4+ cells
    """   
    dC1_dt = k1*C1*T - mu_1*C1
    return dC1_dt

def dixon_dC2_dt(k2, C2, T, mu_2, sigma, I, g):
    """
    Equation for CD8+ and NK cells.
    k2: stimulation rate of CD8+ and NK cells by tumor cells
    C2: concentration of CD8+ and NK cells
    T: concentration of tumor cells
    mu_2: death rate of CD8+ and NK cells
    sigma: maximum proliferation rate of CD8+ by IL-2
    I: concentration of IL-2
    g: half-saturation constant for IL-2 stimulation
    """   
    dC2_dt = k2*C2*T + (sigma*C2*I)/(g + I) - mu_2*C2
    return dC2_dt

def dixon_dT_dt(T, alpha, C2):
    """
    Equation for tumor cells.
    T: concentration of tumor cells
    alpha: killing rate of tumor cells by CD8+ and NK cells
    C2: concentration of CD8+ and NK cells
    """   
    dT_dt = T*(1 - T) - alpha*C2*T
    return dT_dt


def dixon_growth_function(growth_function): 
    """
    Growth function.
    (unused)
    """
    if growth_function == 1:  # linear
        growth = 0.18     
   
    else:
        raise ValueError("Invalid tumor growth function.")  
    
    return growth


def dixon_coupled(t, state, s, beta, mu_0, k1, mu_1, k2, mu_2, sigma, g, alpha):
    """
    Coupled Dixon model equations.
    state: [C1, C2, I, T]
    params: [s, beta, mu_0, k1, mu_1, k2, mu_2, sigma, g, alpha]
    """
    # Unpack state vector
    C1, C2, I, T = state

    # Compute derivatives for each component of the state vector
    dI_dt = dixon_dI_dt(beta, C1, I, mu_0, s)
    dC1_dt = dixon_dC1_dt(k1, C1, T, mu_1)
    dC2_dt = dixon_dC2_dt(k2, C2, T, mu_2, sigma, I, g)
    dT_dt = dixon_dT_dt(T, alpha, C2)

    return [dC1_dt, dC2_dt, dI_dt, dT_dt]


def dixon_coupled_log(t, state, s, beta, mu_0, k1, mu_1, k2, mu_2, sigma, g, alpha,
                      *, active=(True, True, True)):
    """
    The same equations in [log(C1), log(C2), I, log(T)] coordinates.

    active marks initially positive [C1, C2, T]. Zero populations stay zero.
    Cap trial T at its invariant bound while an implicit solver iterates;
    this extension avoids exp overflow outside the physical domain.
    """
    c1, c2, tumor = np.where(active, np.exp([state[0], state[1], min(state[3], 0)]), 0)
    il = state[2]
    return [k1*tumor - mu_1 if active[0] else 0,
            k2*tumor + sigma*il/(g + il) - mu_2 if active[1] else 0,
            beta*c1 - mu_0*il + s,
            1 - tumor - alpha*c2 if active[2] else 0]



def jacobian_log_dixon(t, state, s, beta, mu_0, k1, mu_1, k2, mu_2, sigma, g, alpha,
                       *, active=(True, True, True)):
    """
    Analytical Jacobian for the logarithmic Dixon equations.
    """
    c1, c2, tumor = np.where(active, np.exp([state[0], state[1], min(state[3], 0)]), 0)
    dt_dlogt = tumor if state[3] <= 0 else 0
    jacobian = np.zeros((4, 4))
    if active[0]:
        jacobian[0, 3] = k1*dt_dlogt
    if active[1]:
        jacobian[1, 3] = k2*dt_dlogt
        jacobian[1, 2] = sigma*g/(g + state[2])**2
    jacobian[2, 0] = beta*c1
    jacobian[2, 2] = -mu_0
    if active[2]:
        jacobian[3, 1] = -alpha*c2
        jacobian[3, 3] = -dt_dlogt
    return jacobian
