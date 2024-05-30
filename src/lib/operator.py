#!/usr/bin/env python3
# Author: Taylor C.
"""
Torch-based NUFFT measurement operator for non-uniform Fourier sampling pattern, 
can be applied to both radio-interferometric (RI) and magnetic resonance imaging (MRI) data.
"""

###################################################
# imports

from torchkbnufft import KbNufft, KbNufftAdjoint, calc_tensor_spmatrix
import torch

###################################################

class Operator:
    r""" 
    NUFFT operator class to compute forward :math:`\Phi` and adjoint measurement operator :math:`\Phi^\dagger`
    with associated non-uniform Fourier sampling pattern. Functionalities for backprojection, Point-Spread Function (PSF) 
    computation, residual computation and spectral norm of the operator are also included.
    
    The discrete data acquisition model is defined as
    .. math::
    y = \Phi x + \tau n,
    where :math:`y` is the complex measurement data , :math:`x` is the image of interest, and :math:`n` is the Gaussian
    random noise in the data domain.
    
    """
    def __init__(self, im_size: tuple, op_type: str = 'table', device: torch.device = torch.device('cpu')):
        """
        Initializes the NUFFT operator class.

        :param im_size: dimensions of the image of interest in x and y direction (H, W).
        :type im_size: tuple
        :param op_type: type of interpolation to be used, defaults to 'table', precomputed sparse matrix for min-max interpolation is available as 'sparse_matrix'.
        :type op_type: str, optional
        :param device: device to compute on, defaults to torch.device('cpu'), set to torch.device('cuda') for GPU.
        :type device: torch.device, optional
        
        :raises AssertionError: im_size should be of HxW (tuple of length 2)
        """        

        assert len(im_size) == 2, "im_size should be of HxW"
        self.im_size = im_size
        self.device = device
        self.op_type = op_type
        self.grid_size = (self.im_size[0]*2, self.im_size[1]*2)
        self.forwardOp = KbNufft(im_size=self.im_size,
                                 grid_size = self.grid_size,
                                 numpoints=7).to(self.device)
        self.adjointOp = KbNufftAdjoint(im_size=self.im_size,
                                        grid_size = self.grid_size,
                                        numpoints=7).to(self.device)
        self.uv = None
        self.imweight = None
        self.PSF_peak_val = None
        self.interp_mat = None
        
    def validate_exact(self):
        """
        Validate that sampling pattern is provided. If imweight is not provided, it will be set to 1.
        """
        assert self.uv is not None , 'Sampling pattern must be provided!'
        if self.imweight is None:
            self.imweight = 1.
        
    def set_uv_imweight(self, uv : torch.tensor, imweight : torch.tensor):
        """
        Set the Fourier sampling pattern and weighting to be applied to the measurement. If the operator type is
        'sparse_matrix', the interpolation matrix will be computed.

        :param uv: Fourier sampling pattern, shape (B, 2, N).
        :type uv: torch.tensor
        :param imweight: Weighting to be applied to the measurement, shape (B, 1, N).
        :type imweight: torch.tensor
        """
        # TODO: add assertion for uv and imweight shape
        self.uv = None
        self.imweight = None
        self.uv = uv.to(self.device)
        self.imweight = imweight.to(self.device)
        if self.op_type == 'sparse_matrix':
            self.compute_interp_mat()
        
    def compute_interp_mat(self):
        """
        Compute the interpolation matrix for the given Fourier sampling pattern and image size.

        :raises AssertionError: operator type (op_type) must be `sparse_matrix` to use this function!
        :return: interpolation matrix for the given Fourier sampling pattern and image size.
        :rtype" torch.tensor
            
        """
        assert self.op_type == 'sparse_matrix', 'operator type (op_type) must be `sparse_matrix` to use this function!'
        with torch.no_grad():
            self.interp_mat = calc_tensor_spmatrix(self.uv.squeeze(), 
                                                   im_size=self.im_size, 
                                                   grid_size=self.grid_size, 
                                                   numpoints=7)
        
    def A(self, x : torch.tensor, 
          tau : torch.tensor = torch.tensor([0.]), 
          return_noise : bool = False,
          time_vector : torch.tensor = None):
        """
        Forward measurement operator.
            
        :param x: image(s) for the forward operator to be applied on, shape (B, C, H, W).
        :type x: torch.tensor
        :param tau: noise to be added to the measurement, defaults to 0.
        :type tau: float, optional
        :param return_noise: set True to return the exact noise added to the data, defaults to False.
        :type return_noise: bool, optional
        :return: complex measurement data associated to the image(s) of interest and forward operator, shape (B, 1, N).
        :rtype: torch.tensor
        """
        if tau.size(0) > 1:
            assert tau.size(0) == x.size(0), "tau should have the same batch size as x (in first dimension)"
        x = x.to(self.device)
        x_in = x.to(torch.complex64)
        y = self.forwardOp(x_in, self.uv, self.interp_mat)
        if (tau > 0).all():
            noise = self.noise(y, tau)
        else:
            noise = 0    
        y += noise
        y *= self.imweight
        if return_noise:
            return y, noise
        else:
            return y
    
    def noise(self, y, tau):
        r"""
        Add random complex Gaussian noise from :math:`\mathcal{N}(0, \tau)` to the measurement.

        :param y: measurement to be added noise to, shape (B, 1, N).
        :type y: torch.tensor
        :param tau: standard deviation of a Gaussian noise to be added.
        :type tau: float
        
        :return: noisy measurement, shape (B, 1, N).
        :rtype: torch.tensor
        """
        return (torch.randn_like(y) + 1j * torch.randn_like(y)) * tau / torch.sqrt(torch.tensor(2.))
    
    def At(self, y : torch.tensor):
        """
        Adjoint measurement operator.
        
        :param y: measurement to be applied the adjoint operator on, shape (B, 1, N).
        :type y: torch.tensor
        
        :return: resulting image(s) from the adjoint operator, shape (B, C, H, W).
        :rtype: torch.tensor
        """
        return torch.real(self.adjointOp(y * self.imweight, self.uv, self.interp_mat))
    
    def backproj(self, x : torch.tensor, tau : torch.tensor = torch.tensor([0.]), return_noise : bool = False):
        """
        Backprojection to create dirty image from the provided image of interest using exact measurement operator.
        
        :param x: image of interest, shape (B, C, H, W).
        :type x: torch.tensor
        :param tau: standard deviation of a Gaussian distribution of noise to be added to the measurement, defaults to 0.
        :type tau: float, optional
        :param return_noise: set True to return the exact noise added to the data, defaults to False.
        :type return_noise: bool, optional
        
        :return: dirty image, shape (B, C, H, W).
        :rtype: torch.tensor
        """

        self.validate_exact()
        y = self.A(x, tau, return_noise)
        if return_noise:
            dirty = self.At(y[0])
            noise = y[1]
        else:
            dirty = self.At(y)
        # normalize dirty image by PSF peak pixel value
        dirty /= self.PSF_peak()
        if return_noise:
            return dirty, noise
        else:
            return dirty
        
    def backproj_PSF(self, x : torch.tensor, PSF : torch.tensor):
        """
        Backprojection of dirty image from ground truth image by estimation using convolution with PSF.

        :param x: image of interest, shape (B, C, H, W).
        :type x: torch.tensor
        :param PSF: PSF of the measurement operator, shape (B, C, H, W), either 1x or 2x the image size.
        :type PSF: torch.tensor
        
        :return: approximated dirty image, shape (B, C, H, W).
        :rtype: torch.tensor
        """
        x_fft = torch.fft.fft2(x, s=self.grid_size, dim=[-2, -1])
        if PSF.size(-1) // self.im_size[-1] == 2:
            PSF_fft = torch.fft.fft2(PSF, dim=[-2, -1])
        elif PSF.size(-1) // self.im_size[-1] == 1:
            PSF_fft = torch.fft.fft2(PSF, s=self.grid_size, dim=[-2, -1])
        x_dirty = torch.fft.ifft2(x_fft * PSF_fft, dim=[-2, -1])
        return torch.real(x_dirty)
    
    def backproj_data(self, y : torch.tensor, tau : float = 0., return_noise : bool = False):
        """
        Backprojection to create dirty image from the provided measurement of interst using exact measurement operator.

        :param y: measurement to be applied the adjoint operator on, shape (B, 1, N).
        :type y: torch.tensor
        :param tau: standard deviation of a Gaussian distribution of noise to be added to the measurement, defaults to 0.
        :type tau: float, optional
        :param return_noise: set True to return the exact noise added to the data, defaults to False.
        :type return_noise: bool, optional
        
        :return: dirty image, shape (B, C, H, W).
        :rtype: torch.tensor
        """
        self.validate_exact()
        if tau > 0:
            noise = self.noise(y, tau)
            y += noise
        dirty = self.At(y)
        dirty /= self.PSF_peak()
        if return_noise:
            return dirty, noise
        else:
            return dirty
        
    def gen_res(self, dirty : torch.tensor, x : torch.tensor):
        r"""
        Compute the residual dirty image for given dirty image and reconstruction image, mathematically defined as
        .. math::
        r = \textrm{Re}\{\Phi^\dagger y - \Phi^\dagger \Phi x\},
        
        :param dirty: dirty image, shape (B, C, H, W).
        :type dirty: torch.tensor
        :param x: reconstructed image, shape (B, C, H, W).
        :type x: torch.tensor
        
        :return: residual dirty image, shape (B, C, H, W).
        :rtype: torch.tensor
        """
        return dirty - self.backproj(x)
                
    def gen_res_PSF(self, dirty : torch.tensor, x : torch.tensor, PSF : torch.tensor):
        r"""
        Compute the approximated residual dirty image for given dirty image and reconstruction image, 
        mathematically defined as
        .. math::
        r = \textrm{Re}\{\Phi^\dagger y - x \star \textrm{PSF}\},
        
        :param dirty: dirty image, shape (B, C, H, W).
        :type dirty: torch.tensor
        :param x: reconstructed image, shape (B, C, H, W).
        :type x: torch.tensor
        
        :return: residual dirty image, shape (B, C, H, W).
        :rtype: torch.tensor
        """
        # print(f'oversampling: {PSF.size(-1) // self.im_size[-1]}')
        if PSF.size(-1) // self.im_size[-1] == 2:
            start, end = x.size(3), x.size(3)*2
        elif PSF.size(-1) // self.im_size[-1] == 1:
            start, end = int(self.im_size[0]/2), int(self.im_size[0]+self.im_size[0]/2)
        return dirty - self.backproj_PSF(x, PSF=PSF)[..., start:end, start:end]
        
    def gen_PSF(self, oversampling : int = 2, batch_size : int = 1, normalize : bool = False):
        """
        Generate the point spread function (PSF) for the given Fourier sampling pattern and weighting.
        
        :param oversampling: oversampling factor in Fourier domain, defaults to 2.
        :type oversampling: int, optional
        :param batch_size: number of images in the batch
        :type batch_size: int, optional
        :param normalize: set True to normalize the PSF by its peak value, defaults to False.
        :type normalize: bool, optional
        
        :return: PSF for the given Fourier sampling pattern and weighting.
        :rtype: torch.tensor
        """

        self.validate_exact()
        if oversampling > 1:
            dirac_size = (self.im_size[0]*oversampling, self.im_size[1]*oversampling)
            op_PSF = Operator(im_size=dirac_size, device=self.device)
            op_PSF.set_uv_imweight(self.uv, self.imweight)
            A_PSF = op_PSF.A
            At_PSF = op_PSF.At
        else:
            dirac_size = self.im_size
            A_PSF = self.A
            At_PSF = self.At
        dirac_delta = torch.zeros(dirac_size).to(self.device)
        while len(dirac_delta.shape) < 4:
            dirac_delta = dirac_delta.unsqueeze(0)
        if batch_size > 1:
            assert self.uv.shape[0] == batch_size, "uv should have the same batch size as batch_size (in first dimension)"
            dirac_delta = torch.stack([dirac_delta]*batch_size, dim=0)
        dirac_delta[..., dirac_size[0]//2, dirac_size[1]//2] = 1.
        PSF = At_PSF(A_PSF(dirac_delta))
        if normalize:
            return PSF / torch.amax(PSF, dim=(-1, -2), keepdim=True)
        else:
            return PSF
    
    def PSF_peak(self, batch_size=1):
        """
        Obtain the peak value of the unnormalized PSF.
        
        :param batch_size: number of images in the batch, defaults to 1.
        :type batch_size: int, optional
        
        :return: peak of PSF for the batch.
        :rtype: torch.tensor
        """
        
        if self.PSF_peak_val is None:
            PSF = self.gen_PSF(batch_size)
            PSF_peak = torch.amax(PSF, dim=(-1, -2), keepdim=True)
            self.PSF_peak_val = PSF_peak
        return self.PSF_peak_val

    def op_norm(self, tol=1e-4, max_iter=500, verbose=0):
        """
        Compute spectral norm of the measurement operator using power method.
        
        :param tol: tolerance for relative difference on current and previous solution for stopping the algorithm, defaults to 1e-5.
        :type tol: float, optional
        :param max_iter: maximum number of iteration to compute, defaults to 500.
        :type max_iter: int, optional
        :param verbose: set 1 for verbose output, defaults to 0.
        :type verbose: int, optional
        
        :return: computed spectral norm of the operator.
        :rtype: float
        """
        
        self.validate_exact()
        x = torch.randn(self.im_size).unsqueeze(0).unsqueeze(0).to(self.device)
        x /= torch.linalg.norm(x)
        init_val = 1
        for k in range(max_iter):
            x = self.At(self.A(x))
            val = torch.linalg.norm(x)
            rel_var = torch.abs(val - init_val) / init_val
            if verbose > 1:
                print(f'Iter = {k}, norm = {val}')
            if rel_var < max(2e-6, tol):
                break
            init_val = val
            x = x / val
        if verbose > 0:
            print(f'Norm = {val}\n')
            
        return val
        