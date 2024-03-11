import torchkbnufft as tkbn
import torch

class operator:
    """ Radio-interferometric (RI) operator class with the associating forward and adjoint operators.
    Backprojection to create dirty image and residual generation are also included.
    
    """
    def __init__(self, im_size: tuple, op_type: str = 'table', device: torch.device = torch.device('cpu'), op_acc: str = 'exact'):
        """Initializes the RI operator class with the image size and device

        Parameters
        ----------
        im_size : tuple
            (H, W) of the image of interest
        device : torch.device, optional
            Determine to work on cpu or gpu, by default torch.device('cpu')
        Raises
        ------
        AssertionError
            im_size should be of of length 2 (H, W)
        """
        assert len(im_size) == 2, "im_size should be of HxW"
        self.im_size = im_size
        self.device = device
        self.op_type = op_type
        self.grid_size = (self.im_size[0]*2, self.im_size[1]*2)
        self.forwardOp = tkbn.KbNufft(im_size=self.im_size,
                                        grid_size = self.grid_size,
                                        numpoints=7).to(self.device)
        self.adjointOp = tkbn.KbNufftAdjoint(im_size=self.im_size,
                                                grid_size = self.grid_size,
                                                numpoints=7).to(self.device)
        self.op_acc = op_acc
        self.uv = None
        self.imweight = None
        self.PSF_peak_val = None
        
    def validate_exact(self):
        assert self.uv is not None and self.imweight is not None, 'uv and imweight must be provided for exact measurement operator'
        
    def set_uv_imweight(self, uv : torch.tensor, imweight : torch.tensor):
        """Set the Fourier sampling pattern and weighting to be applied to the measurement. If the operator type is
        'sparse_matrix', the interpolation matrix will be computed.
        
        Parameters
        ----------
        uv : torch.Tensor
            Fourier sampling pattern, shape (B, 2, N)
        imweight : torch.Tensor
            Weighting to be applied to the measurement, shape (B, 1, N)
        """
        self.uv = uv
        self.imweight = imweight
        if self.op_type == 'sparse_matrix':
            self.compute_interp_mat()
        else:
            self.interp_mat = None
        
    def compute_interp_mat(self):
        """Compute the interpolation matrix for the given Fourier sampling pattern and image size.

        Returns
        -------
        torch.Tensor
            Interpolation matrix for the given Fourier sampling pattern and image size.
        """
        assert self.op_type == 'sparse_matrix', 'operator type (op_type) must be `sparse_matrix` to use this function!'
        with torch.no_grad():
            self.interp_mat = tkbn.calc_tensor_spmatrix(self.uv.squeeze(), im_size=self.im_size, grid_size=self.grid_size, numpoints=7)
        
        
    def A(self, x, tau=0):
        """Forward operator.

        Parameters
        ----------
        x : torch.Tensor
            Image(s) for the forward operator to be applied on, shape (B, C, H, W)
        tau : float, optional
            Noise to be added to the measurement, by default 0

        Returns
        -------
        torch.Tensor
            Measurement associated to the image(s) of interest and forward operator, shape (B, 1, N)
        """
        x_in = x.to(torch.complex64)
        y = self.forwardOp(x_in, self.uv, self.interp_mat)
        if tau is not None:
            if tau > 0:
                y = self.noise(y, tau)
        return y * self.imweight
    
    def noise(self, y, tau):
        """Add noise to the measurement.

        Parameters
        ----------
        y : torch.Tensor
            Measurement to be added noise to, shape (B, 1, N)
        tau : float
            Standard deviation of a Gaussian noise to be added

        Returns
        -------
        torch.Tensor
            Noisy measurement, shape (B, 1, N)
        """
        return y + (torch.randn_like(y) + 1j * torch.randn_like(y)) * tau / torch.sqrt(torch.tensor(2.))
    
    def At(self, y):
        """Adjoint operator.

        Parameters
        ----------
        y : torch.Tensor
            Measurement to be applied the adjoint operator on, shape (B, 1, N)

        Returns
        -------
        torch.Tensor
            Resulting image(s) from the adjoint operator, shape (B, C, H, W)
        """
        return torch.real(self.adjointOp(y * self.imweight, self.uv, self.interp_mat))
    
    def backproj(self, x, PSF=None, tau=None):
        """Backprojection to create dirty image from the given image, using either exact measurement operator or
        approximation by convolution with PSF.

        Parameters
        ----------
        x : torch.Tensor
            Image of interest, shape (B, C, H, W).
        PSF : torch.Tensor, optional
            PSF of the measurement operator, shape (B, C, H, W). Required if op_acc is 'approx'
        tau : float, optional
            Standard deviation of a Gaussian distribution of noise to be added to the measurement, by default None

        Returns
        -------
        torch.Tensor
            Dirty image.
        """
        if self.op_acc == 'exact':
            self.validate_exact()
            dirty = self.At(self.A(x, tau))
            if self.PSF_peak_val is not None:
                dirty /= self.PSF_peak_val
            else:
                dirty /= self.PSF_peak()
            return dirty
        elif self.op_acc == 'approx':
            assert PSF is not None, 'PSF must be provided for approximated measurement operator'
            x_fft = torch.fft.fft2(x, s=self.grid_size, dim=[-2, -1])
            if PSF.size(-1) // self.im_size[-1] == 2:
                PSF_fft = torch.fft.fft2(PSF, dim=[-2, -1])
            elif PSF.size(-1) // self.im_size[-1] == 1:
                PSF_fft = torch.fft.fft2(PSF, s=self.grid_size, dim=[-2, -1])
            x_dirty = torch.fft.ifft2(x_fft * PSF_fft, dim=[-2, -1])
            return torch.real(x_dirty)
    
    def backproj_data(self, y):
        """Backprojection to create dirty image from the given image, using either exact measurement operator or
        approximation by convolution with PSF.

        Parameters
        ----------
        x : torch.Tensor
            Image of interest, shape (B, C, H, W).
        PSF : torch.Tensor, optional
            PSF of the measurement operator, shape (B, C, H, W). Required if op_acc is 'approx'
        tau : float, optional
            Standard deviation of a Gaussian distribution of noise to be added to the measurement, by default None

        Returns
        -------
        torch.Tensor
            Dirty image.
        """
        assert self.op_acc == 'exact', 'op_acc must be exact for backproj_data'
        self.validate_exact()
        dirty = self.At(y)
        if self.PSF_peak_val is not None:
            dirty /= self.PSF_peak_val
        else:
            dirty /= self.PSF_peak()
        return dirty
        
    def gen_res(self, dirty, x, PSF=None):
        """Compute the residual dirty image for given dirty image and reconstruction image, using
        either exact measurement operator or approximation by convolution with PSF.

        Parameters
        ----------
        dirty : torch.Tensor
            Dirty image, shape (B, C, H, W).
        x : torch.Tensor
            Reconstruction image, shape (B, C, H, W).
        PSF : torch.Tensor, optional
            PSF of the measurement operator, shape (B, C, H, W). Required if op_acc is 'approx'

        Returns
        -------
        torch.Tensor
            Residual dirty image.
        """
        if self.op_acc == 'exact':
            self.validate_exact()
            return dirty - self.backproj(x)
        elif self.op_acc == 'approx':
            if PSF.size(-1) // self.im_size[-1] == 2:
                start, end = x.size(3), x.size(3)*2
            elif PSF.size(-1) // self.im_size[-1] == 1:
                start, end = int(self.im_size[0]/2), int(self.im_size[0]+self.im_size[0]/2)
            return dirty - self.backproj(x, PSF=PSF)[..., start:end, start:end]
        
    def gen_PSF(self, batch_size=1, normalize=False):
        """Generate the point spread function (PSF) for the given Fourier sampling pattern and weighting.

        Parameters
        ----------
        os: int, optional
            Oversampling factor, by default 1.
        batch_size : int, optional
            Number of images in the batch, by default 1.

        Returns
        -------
        torch.Tensor
            PSF for the given Fourier sampling pattern and weighting.
        """
        self.validate_exact()
        # if os > 1:
        #     dirac_size = (self.im_size[0]*os, self.im_size[1]*os)
        #     op_PSF = operator(im_size=dirac_size, device=self.device, op_acc='exact')
        #     A_PSF = op_PSF.A
        #     At_PSF = op_PSF.At
        # else:
        #     dirac_size = self.im_size
        #     A_PSF = self.A
        #     At_PSF = self.At
        dirac_delta = torch.zeros(self.im_size).to(self.device)
        while len(dirac_delta.shape) < 4:
            dirac_delta = dirac_delta.unsqueeze(0)
        if batch_size > 1:
            assert self.uv.shape[0] == batch_size, "uv should have the same batch size as batch_size (in first dimension)"
            dirac_delta = torch.stack([dirac_delta]*batch_size, dim=0)
        dirac_delta[..., self.im_size[0]//2, self.im_size[1]//2] = 1.
        PSF = self.At(self.A(dirac_delta))
        if normalize:
            return PSF / torch.amax(PSF, dim=(-1, -2), keepdim=True)
        else:
            return PSF
    
    def PSF_peak(self, batch_size=1):
        """Compute the peak of PSF.

        Parameters
        ----------
        batch_size : int, optional
            Number of images in the batch, by default 1.

        Returns
        -------
        torch.Tensor
            A tensor containing the peak of PSF for the batch.
        """
        PSF = self.gen_PSF(batch_size)
        PSF_peak = torch.amax(PSF, dim=(-1, -2), keepdim=True)
        if self.PSF_peak_val is None:
            self.PSF_peak_val = PSF_peak
        return PSF_peak

    def op_norm(self, tol=1e-4, max_iter=500, verbose=0):
        """Compute spectral norm of the operator.

        Parameters
        ----------
        tol : float, optional
            Tolerance for relative difference on current and previous solution for stopping the algorithm, by default 1e-5.
        max_iter : int, optional
            Maximum number of iteration to compute, by default 500.
        verbose : int, optional
            By default 0.

        Returns
        -------
        float
            The computed spectral norm of the operator.
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
        