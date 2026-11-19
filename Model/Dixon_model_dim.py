# Dixon model (dimensionalized)
# This is another immunotherapy model involving IL-2, CD4+, CD8+ and NK cells.
# the Dixon model is unused but the project could be expanded to include it later.

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

def dixon_dT_dt(T, alpha, C2, sigma, K):
    """
    Equation for tumor cells.
    T: concentration of tumor cells
    alpha: killing rate of tumor cells by CD8+ and NK cells
    C2: concentration of CD8+ and NK cells
    sigma: per capita growth rate of tumor cells
    K: carrying capacity of the tumor cells

    """   
    dT_dt = sigma*T*(1 - T/K) - alpha*C2*T
    return dT_dt


def dixon_coupled(t, state, beta, C1, I, mu_0, s, k1, mu_1, k2, mu_2, sigma, g, alpha, K):
    """
    Coupled Dixon model equations.
    state: [I, C1, C2, T]
    """
    # Unpack state vector
    I, C1, C2, T = state

    # Compute derivatives for each component of the state vector
    dI_dt = dixon_dI_dt(beta, C1, I, mu_0, s)
    dC1_dt = dixon_dC1_dt(k1, C1, T, mu_1)
    dC2_dt = dixon_dC2_dt(k2, C2, T, mu_2, sigma, I, g)
    dT_dt = dixon_dT_dt(T, alpha, C2, sigma, K)

    return [dI_dt, dC1_dt, dC2_dt, dT_dt]


def dixon_dimensionalize(t, lamb, beta, C1, C2, T, I, mu_0, s, k1, mu_1, k2, mu_2, sigma, g, alpha, K, tau_1, tau_2):
    """
    Dimensionalize the Dixon model state vector.
    t: time
    lamb: scaling parameter for time
    beta: stimulation rate of IL-2 by CD4+ cells
    C1: concentration of CD4+ cells
    C2: concentration of CD8+ and NK cells
    T: concentration of tumor cells
    I: concentration of IL-2
    mu_0: decay rate of IL-2
    s: treatment term or external source of IL-2
    k1: stimulation rate of CD4+ cells by tumor cells
    mu_1: death rate of CD4+ cells
    k2: stimulation rate of CD8+ and NK cells by tumor cells
    mu_2: death rate of CD8+ and NK cells
    sigma: maximum proliferation rate of CD8+ by IL-2
    g: half-saturation constant for IL-2 stimulation
    alpha: killing rate of tumor cells by CD8+ and NK cells
    K: carrying capacity of the tumor cells
    tau_1: parameter tau_1
    tau_2: parameter tau_2
    """
    I_bar = I * K
    C1_bar = C1 * K
    C2_bar = C2 * K
    T_bar = T * K

    t_bar = t / lamb
    sigma_bar = sigma * lamb

    g_bar = g * K

    k1_bar = k1 * lamb / K
    k2_bar = k2 * lamb / K

    s_bar = s * K * lamb

    alpha_bar = alpha * lamb / K

    mu_0_bar = mu_0 * lamb
    mu_1_bar = mu_1 * lamb
    mu_2_bar = mu_2 * lamb

    beta_bar = beta * lamb

    tau_1_bar = tau_1 * lamb / K
    tau_2_bar = tau_2 * lamb 

    dimensionalized_parameters = [I_bar, C1_bar, C2_bar, T_bar, t_bar, sigma_bar, g_bar, k1_bar, k2_bar, s_bar, alpha_bar, mu_0_bar, mu_1_bar, mu_2_bar, beta_bar, tau_1_bar, tau_2_bar]

    return dimensionalized_parameters