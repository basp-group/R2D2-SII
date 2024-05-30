import torch
import numpy as np


def gen_imaging_weights(u, 
                        v, 
                        nW, 
                        im_size, 
                        weight_type='robust', 
                        weight_gridsize=1.2, 
                        weight_robustness=0.0):
    """
    Parameters
    ----------
    u : torch.tensor
        u coordinate of the data point in the Fourier domain in radians.
    v : torch.tensor
        v coordinate of the data points in the Fourier domain in radians.
    nW : torch.tensor
        square root of inverse of the variance.
    im_size:  tuple
        image dimension
    weight_type : str
        Type of weight: {'uniform', 'robust', 'none'}. Default is 'robust'.
    weight_gridsize : int
        Grid size for weight calculation. Default is 1.
    weight_robustness : float
        The Briggs parameter for robust weighting. Default is 0.0.

    Returns
    -------
    nWimag : torch.tensor
        weights inferred from the density of the sampling (uniform/Briggs).
    """

    nmeas = u.numel()
    N = torch.tensor([np.floor(i * weight_gridsize).astype(int) for i in im_size])
    nW2 = (nW.view(-1).to(u.device))**2
        
    if nW2.size(-1) == 1:
        nW2 = nW2 * torch.ones(nmeas, device=u.device)
    u[v < 0] = -u[v < 0]
    v[v < 0] = -v[v < 0]
    
    # Initialize gridded weights matrix with zeros
    p = torch.floor((v + torch.pi) * N[0] / 2 / torch.pi).to(torch.int64).view(-1) - 1
    q = torch.floor((u + torch.pi) * N[1] / 2 / torch.pi).to(torch.int64).view(-1) - 1
    gridded_weights = torch.zeros(torch.prod(N), dtype=torch.float64, device=u.device)
    uvInd = p * N[1] + q
    if weight_type != 'none':
        if weight_type == 'uniform':
            values = torch.ones_like(p, dtype=torch.float64)
        elif weight_type == 'briggs':
            values = nW2

    # Use scatter_add_ to update gridded_weights
    gridded_weights.scatter_add_(0, uvInd, values.to(torch.float64))

    # Apply weighting based on weighting_type
    if weight_type != 'none':
        gridded_vals = gridded_weights[uvInd]  # Gather the values

        if weight_type == 'uniform':
            nWimag = 1 / torch.sqrt(gridded_vals)
        elif weight_type == 'briggs':
            # Compute robust scale factor
            robust_scale = (torch.sum(nW2) / torch.sum(gridded_weights ** 2)) * (5 * 10 ** (-weight_robustness)) ** 2
            nWimag = 1 / torch.sqrt(1 + robust_scale * gridded_vals)    
    return nWimag.view(1, 1, -1)