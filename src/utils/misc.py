from scipy.optimize import fsolve

def vprint(message, verbose_lv, verbose):
    if verbose_lv >= verbose:
        print(message)
        

def solve_expo_factor(sigma_0, sigma):
    """Compute exponentation factor for given sigma0 and sigma values.

    Parameters
    ----------
    sigma_0 : float
        1/ current dynamic range of the image of interest.
    sigma : float
        1/ target dynamic range
    Returns
    -------
    float
        Exponentiation factor.
    """
    fun = lambda a: (1 + a * sigma) ** (1 / sigma_0) - a

    est_c = sigma ** -(1 / (1 / sigma_0 - 1))
    est_a = (est_c - 1) / sigma

    res = fsolve(fun, est_a)
    obj = fun(res)

    if obj > 1e-7 or res < 40:
        print(f'Possible wrong solution. sigma = {sigma}, a = {res[0]}, f(a) = {obj[0]}')
    return res[0]