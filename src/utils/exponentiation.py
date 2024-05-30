import torch
import numpy as np
from scipy.optimize import fsolve

def expo_im(x, a) -> torch.tensor:
    r"""
    Exponentiate the image of interest by factor :math:`a` defined as
    
    .. math::
    x_{\text{exp}} = \frac{a^{x} - 1}{a},
        
    where :math:`x` is the image of interest and :math:`x_{\text{exp}}` is the exponentiated image.
    
    :param x: image of interest
    :type x: torch.Tensor
    :param a: exponentiation factor
    :type a: float
    
    :return: exponentiated image
    :rtype: torch.Tensor
    """
    return (a**x - 1) / a

def solve_expo_factor(sigma_0, sigma) -> float:
    r"""
    Compute exponentation factor :math:`a` to exponentiate the image of interest
    from current dynamic range :math:`1/\sigma_0` to target dynamic range :math:`1/\sigma` by solving
    
    .. math::
    a = (1 + a \sigma)^{\sigma_0^{-1}}.
    
    :param sigma_0: inverse of current dynamic range of the image of interest
    :type sigma_0: float
    :param sigma: inverse of target dynamic range.
    :type sigma: float
    
    :return: exponentiation factor.
    :rtype: float
    """
    fun = lambda a: (1 + a * sigma) ** (1 / sigma_0) - a

    est_c = sigma ** -(1 / (1 / sigma_0 - 1))
    est_a = (est_c - 1) / sigma

    res = fsolve(fun, est_a)
    obj = fun(res)

    if obj > 1e-7 or res < 40:
        print(f'Possible wrong solution. sigma = {sigma}, a = {res[0]}, f(a) = {obj[0]}')
    return res[0]

def compute_tau(op,
                data_dict : dict,
                sigma_range_min : float = 2e-6,
                sigma_range_max : float = 1e-3,
                briggs : bool = False,
                return_sub_op : bool = False,
                device : torch.device = torch.device('cpu')) -> tuple:
    assert sigma_range_min > 0 and sigma_range_max > 0, 'sigma_range_min and sigma_range_max must be positive'
    assert sigma_range_min < sigma_range_max, 'sigma_range_min must be smaller than sigma_range_max'
    log_sigma = (np.random.rand(1, 1) * (np.log10(sigma_range_max) - np.log10(sigma_range_min))
                    + np.log10(sigma_range_min))
    sigma = (10 ** log_sigma)[0][0]
    sigma = torch.tensor(sigma).to(device)
    if 'time' not in data_dict:
        tau = compute_tau_instance(op, data_dict, sigma, briggs)
    else:
        print('Time series data is detected. Computing tau for each time instance')
        time = data_dict['time'].squeeze()
        time_diff = time[1:] - time[:-1]
        new_start = torch.cat([torch.tensor([0]).to(device), torch.where(abs(time_diff) > 50)[0] + 1, torch.tensor([max(time.shape)]).to(device)])
        # new_start = torch.cat([torch.tensor([0]).to(device), torch.where(time_diff < 0)[0] + 1, torch.tensor([max(time.shape)]).to(device)])
        tau = torch.zeros(max(data_dict['uv'].shape), device=device)
        print(f'Number of time instances: {len(new_start) - 1}')
        for i in range(len(new_start) - 1):
            print(f'Computing tau for time instance {i + 1}/{len(new_start) - 1}: start: {new_start[i]}, end: {new_start[i + 1]}')
            uv = data_dict['uv'][..., new_start[i]:new_start[i + 1]]
            data_dict_instance = {'uv': uv}
            if 'nWimag' in data_dict and data_dict['nWimag'].numel() > 1:
                nWimag = data_dict['nWimag'][..., new_start[i]:new_start[i + 1]]
                data_dict_instance.update({'nWimag': nWimag})
            if 'nW' in data_dict  and data_dict['nW'].numel() > 1:
                nW = data_dict['nW'][..., new_start[i]:new_start[i + 1]]
                data_dict_instance.update({'nW': nW})
            tau_tmp = compute_tau_instance(op, data_dict_instance, sigma, briggs)
            print(f'{i+1}: tau= {tau_tmp}')
            tau[new_start[i]:new_start[i + 1]] = tau_tmp
    
    if len(tau.size()) == 0 or len(tau.shape) == 1:
        tau = tau.unsqueeze(0)
    if len(sigma.size()) == 0:
        sigma = sigma.unsqueeze(0)
    if return_sub_op:
        return tau, sigma, new_start
    else:
        return tau, sigma

def compute_tau_instance(op, 
                         data_dict : dict, 
                         sigma : torch.tensor,
                         briggs : bool = False) -> tuple:
    r"""
    Compute the standard deviation of the heuristic noise level in the measurement defined as
    
    .. math::
    \tau = \sigma \sqrt{2 \|\Phi\|_S},
    
    where :math:`\sigma` is the standard deviation of noise in the image domain 
    and :math:`\|\Phi\|_S` is the spectral norm of the measurement operator :math:`\Phi`.

    :param op: measurement operator
    :type op: lib.operator.Operator
    :param data_dict: dictionary containing uv points and imweight weighting from uv data file
    :type data_dict: dict
    :param sigma_range_min: minimum value of :math:`\sigma`, defaults to 2e-6
    :type sigma_range_min: float, optional
    :param sigma_range_max: maximum value of :math:`\sigma`, defaults to 1e-3
    :type sigma_range_max: float, optional
    :param briggs: set briggs weighting, defaults to False
    :type briggs: bool, optional
    :param device: target device for computation of torch tensor, defaults to torch.device('cpu')
    :type device: torch.device, optional
    
    :return: tuple of :math:`\tau` and :math:`\sigma`, noise level in the data domain and standard deviation of noise in the image domain
    :rtype: tuple
    
    :raises AssertionError: if sigma_range_min or sigma_range_max is not positive or sigma_range_min is greater than sigma_range_max
    """
    weight = torch.ones(data_dict['uv'].shape[-1], device=data_dict['uv'].device).unsqueeze(0).unsqueeze(0)
    weight2 = torch.ones(data_dict['uv'].shape[-1], device=data_dict['uv'].device).unsqueeze(0).unsqueeze(0)
    if 'nWimag' in data_dict:
        weight *= data_dict['nWimag']
        weight2 = weight * data_dict['nWimag']
    else:
        print('No nWimag found for computing tau.')
    # if 'nW' in data_dict:
    #     weight *= data_dict['nW']
    op.set_uv_imweight(data_dict['uv'], weight)
    op_norm = op.op_norm()
    print(op_norm.item())
    tau = sigma * torch.sqrt(2 * op_norm)
    if briggs:
        op.set_uv_imweight(data_dict['uv'], weight2)
        op_norm_double = op.op_norm()
        tau = tau / torch.sqrt(op_norm_double / op_norm)
    return tau