# Dixon model
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
