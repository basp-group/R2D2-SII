import argparse
import logging
import pathlib
import timeit
import os
import torch
import numpy as np
from pathlib import Path
from astropy.io import fits
from scipy.io import loadmat
from torch.utils.data import Dataset

from lib.operator import Operator
from utils.gen_imaging_weights import gen_imaging_weights

def remove_lightning_console_log():
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

def save_reconstructions(reconstructions, out_dir, rec_file_ext='_rec', res_file_ext='_dirty'):
    """Saves the reconstructions of the test dataset into fits files.

    Parameters
    ----------
    reconstructions : dict[str, np.array]
        A dictionary mapping input filenames to corresponding reconstructions.\
    out_dir : str
        Path to the output directory where the reconstructions should be saved.
    rec_file_ext : str, optional
        Extension for the reconstructed image file to be saved, by default '_rec'
    res_file_ext : str, optional
        Extension for the residual dirty image file, by default '_dirty'
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    for fname, recons in reconstructions.items():
        fname = fname.split('/')[-1]
        fits.writeto(os.path.join(out_dir, fname.split(res_file_ext)[0], f'{rec_file_ext}.fits'), recons[0][1].squeeze().cpu().numpy(), overwrite=True)
        
def read_fits_as_tensor(path):
    x = fits.getdata(path)
    x = torch.tensor(x.astype(np.float64)).unsqueeze(0).unsqueeze(0)
    return x

def read_uv(args: argparse.Namespace,
            im_size: tuple,
            device=torch.device('cpu'), 
            multi_noise : bool = False):
    """Read u, v and imweight from specified path.

    Parameters
    ----------
    uv_path : str
        Path to the file containing u, v and imweight.
    fname_uv : str
        Filename of the uv file.
    imweight_name : str, optional
        Specific name of variable containing the weighting to be used in the uv file, by default 'nWimag'.
    device : torch.device, optional
        Device to be used, by default torch.device('cpu')

    Returns
    -------
    uv: torch.Tensor
        Fourier sampling pattern, shape (2, N)
    imweight : torch.Tensor
            Weighting to be applied to the measurement, shape (1, N)
    """
    
    # load provided uv_file_path
    uv_file = loadmat(args.data_file)
    if 'super_resolution' in uv_file and uv_file['super_resolution'].item() != super_resolution:
        print(f'WARNING: super resolution in the uv file ({uv_file["super_resolution"].item()}) does not match the provided super resolution ({super_resolution}).')
        print(f'WARNING: overwriting specified super resolution with the one found in the uv file.')
        super_resolution = uv_file['super_resolution'].item()
    else:
        super_resolution = args.super_resolution
    
    # read u and v
    u = uv_file['u'].astype(np.float64).squeeze()
    v = uv_file['v'].astype(np.float64).squeeze()
    uv = torch.tensor(np.stack((-v, u), axis=1)).unsqueeze(0)
    if uv.size(-1) == 2:
        uv = uv.permute(0, 2, 1)
        
    # if uv coordinates are in meters, convert to wavelength
    if 'unit' in uv_file:
        if uv_file['unit'].item() == 'm':
            print('INFO: converting uv coordinate unit from meters to wavelength.')
            from scipy.constants import c
            wavelength = c / uv_file['frequency'].item()
            uv = uv / wavelength
    
    max_proj_baseline = torch.sqrt(torch.max(uv[:, 0,:]**2 + uv[:, 1,:]**2)).item()
    
    # convert uv coordinates to radians
    uv = uv * torch.pi / (super_resolution * max_proj_baseline)
    uv = uv.to(device)
    data_dict = {'uv': uv}
    try:
        if uv_file['nW'].shape[-1] == 1 or uv_file['nW'].shape[-1] == uv_file['u'].shape[-1]:
            nW = uv_file['nW']
        else:
            tau_index, nW_unique = zip(*sorted(zip(uv_file['tau_index'].squeeze(), uv_file['nW'].squeeze())))
            tau_index = tau_index + (max(uv_file['u'].shape),)
            nW = np.zeros(max(uv_file['u'].shape))
            for i in range(len(tau_index)-1):
                nW[tau_index[i]:tau_index[i+1]] = nW_unique[i]
        nW = torch.tensor(nW.squeeze().astype(np.float32))
        while nW.dim() < 3:
            nW = nW.unsqueeze(0)
        print('INFO: using provided nW.')
        
    except:
        # raise ValueError(f'Imaging weights nW not found in the uv file.')
        nW = torch.tensor([1])
    nW = nW.to(device)
    data_dict.update({'nW': nW})
    
    # read image weighting according to specified imweight_name (nWimag by default)
    if not args.gen_nWimag:
        try:
            print(f'INFO: using provided imaging weights "{args.imweight_name}".')
            nWimag = torch.tensor(uv_file[args.imweight_name].astype(np.float32).squeeze()).unsqueeze(0).unsqueeze(0)
        except:
            raise ValueError(f'Imaging weights {args.imweight_name} not found in the uv file.')
    else:
        print("Generating imaging weights ...")
        print(f'Weight type: {args.weight_type}')
        if args.weight_type == "briggs":
            if 'weight_robustness' not in data_dict:
                match args.weight_robustness:
                    case "random":
                        weight_robustness = (
                            args.weight_robustness_min
                            + (args.weight_robustness_max - args.weight_robustness_min) * np.random.uniform()
                        )
                    case "zero":
                        weight_robustness = 0.0
            else:
                weight_robustness = data_dict["weight_robustness"]
            print(f'Weight robustness: {weight_robustness}')
        else:
            weight_robustness = None
        nWimag = gen_imaging_weights(
            u=data_dict["uv"][0, 1, :].clone(), 
            v=-data_dict["uv"][0, 0, :].clone(), 
            nW=data_dict["nW"], 
            im_size=im_size, 
            weight_type=args.weight_type, 
            weight_gridsize=args.weight_gridsize, 
            weight_robustness=weight_robustness
        )
    nWimag = nWimag.to(device)
    data_dict.update({'nWimag': nWimag})
   
        
    if multi_noise:
        assert 'time' in uv_file, 'time vector not found in the uv file.'
        data_dict.update({'time': torch.tensor(uv_file['time'].squeeze()).to(device)})
    
    return data_dict

def get_data(args, device, res_device, filename):
    """Read data from paths specified in args and generate the measurement operator and its adjoint.
    All data are converted to torch.Tensor, dirty and ground truth images will be normalized by the mean of dirty. 

    Parameters
    ----------
    args : _ArgumentParser
        Arguments parsed from command line and processed.

    Returns
    -------
    tuple
        Tuple containing the data (dict), measurement operator (operator class), uv coordinates (tensors) and imweight (tensors).
    """
    im_size = (args.im_dim_x, args.im_dim_y)
    op = Operator(im_size=im_size, op_type=args.operator_type, device=res_device)
    if str(args.data_file).endswith('.fits'):
        dirty = read_fits_as_tensor(args.data_file).to(device)
        assert args.uv_file is not None, "uv_file must be provided for .fits data."
        assert args.uv_file.endswith('.mat'), 'uv file must end format.'
        data_dict = read_uv(args, device=res_device)
        dirty_time = 0
        op.set_uv_imweight(data_dict['uv'], data_dict['nWimag'])
    elif str(args.data_file).endswith('.mat'):
        data = loadmat(args.data_file)
        for var in ['y', 'u', 'v', 'nWimag', 'nW']:
            assert var in data, f'{var} not found in data_file.'
        data_dict = read_uv(args,
                            im_size=im_size,
                            device=res_device)
        uv = data_dict['uv'].to(torch.float32)
        nW = data_dict['nW'].to(torch.float32)
        nWimag = data_dict['nWimag'].to(torch.float32)
        if args.natural_weight:
            nWimag *= nW
        
        # apply flagging to data, uv coordinates and nWimag
        if nW.size(-1) > 1:
            baselines = torch.tensor(np.sqrt(data['u'].squeeze()**2 + data['v'].squeeze()**2)).to(res_device)
            flag = (baselines>0) * (nW.squeeze()>0)
            if flag.sum() < len(flag):
                print(f'INFO: flagging {len(flag) - flag.sum()} visibilities.')
                y = y[..., flag]
                uv = uv[..., flag]
                nWimag = nWimag[..., flag]
        op.set_uv_imweight(uv, nWimag)
        
        # read visibilities
        y = data['y'].squeeze()
        y = torch.tensor(y).unsqueeze(0).unsqueeze(0).to(res_device).to(torch.complex64)
        
        # compute dirty image and save to specified output path
        ts = timeit.default_timer()
        dirty = op.backproj_data(y * nWimag).to(device)
        dirty_time = timeit.default_timer() - ts
        print(f'INFO: time to compute dirty image (normalized backprojected data): {dirty_time:.6f} sec')
        fits.writeto(os.path.join(args.output_path, 'dirty.fits'), 
                     dirty.clone().squeeze().numpy(force=True), 
                     overwrite=True)
    
    # read and normalize ground truth image if provided
    if args.gdth_file is not None:
        gdth = read_fits_as_tensor(args.gdth_file).to(device)
    else:
        gdth = 0.
    
    data = {'dirty': dirty, 
             'gdth': gdth, 
             'uv': uv,
             'nWimag': nWimag,
             'fname': filename}
    return data, op, dirty_time

class Data_N1(Dataset):
    """
    A PyTorch Dataset that provides access to image slices.
    """

    def __init__(self, hparams, data_partition, transform):
        self.transform = transform
        self.dirty = []
        self.clean = []
        self.mat_files = []
        self.hparams = hparams
        root = os.path.join(self.hparams.data_path, f'{data_partition}') # ground truth
        root2 = os.path.join(self.hparams.data_path, f'{data_partition}{self.hparams.res_ext}') # dirty/ residual dirty
        mat_path = os.path.join(self.hparams.data_path, f'{data_partition}{self.hparams.mat_ext}') # dynamic range of the exponentiated image (1/a)
        if self.hparams.layers > 1:
            self.PSF = []
            PSF_path = self.hparams.PSF_path
        files_dt = list(pathlib.Path(root2).iterdir())

        for fname in sorted(files_dt):
            dirty_pth = str(fname.resolve())
            im = fits.getdata(dirty_pth)
            im = np.expand_dims(im, 0)
            num_slices = im.shape[0]
            self.dirty += [(dirty_pth, slice) for slice in range(num_slices)]

            clean_fname = (dirty_pth.split('/')[-1]).split('.fits')[0].split(self.hparams.dirty_file_ext)[0]
            if len(self.hparams.gdth_file_ext) > 0:
                clean_pth = os.path.join(root, f'{clean_fname}{self.hparams.gdth_file_ext}.fits')
            else:
                clean_pth = os.path.join(root, f'{clean_fname}.fits')
            im = fits.getdata(clean_pth)
            im = np.expand_dims(im, 0)
            num_slices = im.shape[0]
            self.clean += [(clean_pth, slice) for slice in range(num_slices)]
            
            if self.hparams.layers > 1:
                uv_id = clean_fname.split('_id_')[-1]
                PSF_name = f'uv_id_{uv_id}{self.hparams.PSF_file_ext}.fits'
                PSF_pth = os.path.join(PSF_path, PSF_name)
                im = fits.getdata(PSF_pth)
                im = np.expand_dims(im, 0)
                num_slices = im.shape[0]
                self.PSF += [(PSF_pth, slice) for slice in range(num_slices)]
                
            self.mat_files += [os.path.join(mat_path, f'{clean_fname}{self.hparams.mat_file_ext}.mat')]
        print('Number of dirty images:', len(self.dirty))
        print('Number of ground truth images:', len(self.clean))
        if self.hparams.layers > 1:
            print('Number of PSF images:', len(self.PSF))

    def __len__(self):
        return len(self.dirty)

    def __getitem__(self, i):
        fname, slice = self.clean[i]
        fname_dt, slice_dt = self.dirty[i]

        original_image = np.expand_dims(fits.getdata(fname), axis=0)  # Adding a channel dimension. shape of output is [1, W, H]
        dirty = np.expand_dims(fits.getdata(fname_dt), axis=0)  # Adding a channel dimension. shape of output is [1, W, H]

        if self.hparams.layers > 1:
            fname_PSF, slice_PSF = self.PSF[i]
            PSF = fits.getdata(fname_PSF)
            PSF = np.expand_dims(PSF/PSF.max(), axis=0)
        else:
            PSF = 0.
        try:
            mat_file = loadmat(self.mat_files[i])
            try:    
                a_expo = mat_file['a_expo'].squeeze()
            except:
                try:
                    a_expo = mat_file['expo_factor'].squeeze()
                except:
                    a_expo = 0.
        except:
            a_expo = 0.
        return self.transform(original_image, dirty, PSF, fname_dt.split('/')[-1], slice, a_expo)

class Data_Ni(Dataset):
    """
    A PyTorch Dataset that provides access to image slices.
    """

    def __init__(self, hparams, data_partition, transform):
        self.transform = transform
        self.res = []
        self.clean = []
        self.rec = []
        self.mat_files = []
        self.hparams = hparams
        root = os.path.join(self.hparams.data_path, f'{data_partition}') # ground truth
        root2 = os.path.join(self.hparams.data_path, f'{data_partition}{self.hparams.res_ext}') # residual dirty
        root3 = os.path.join(self.hparams.data_path, f'{data_partition}{self.hparams.rec_ext}') # reconstruction
        mat_path = os.path.join(self.hparams.data_path, f'{data_partition}{self.hparams.mat_ext}') # dynamic range of the exponentiated image (1/a)
        if self.hparams.layers > 1:
            self.PSF = []
            PSF_path = self.hparams.PSF_path
        files_res = list(pathlib.Path(root2).iterdir())

        for fname in sorted(files_res):
            res_pth = str(fname.resolve())
            im = fits.getdata(res_pth)
            im = np.expand_dims(im, 0)
            num_slices = im.shape[0]
            self.res += [(res_pth, slice) for slice in range(num_slices)]

            clean_fname = (res_pth.split('/')[-1]).split('_res')[0]
            
            clean_pth = os.path.join(str(root.resolve()), clean_fname, f'{self.hparams.gdth_file_ext}.fits')
            im = fits.getdata(clean_pth)
            im = np.expand_dims(im, 0)
            num_slices = im.shape[0]
            self.clean += [(clean_pth, slice) for slice in range(num_slices)]

            rec_pth = os.path.join(str(root3.resolve()), clean_fname, f'{self.hparams.rec_file_ext}.fits')
            im = fits.getdata(rec_pth)
            im = np.expand_dims(im, 0)
            num_slices = im.shape[0]
            self.rec += [(rec_pth, slice) for slice in range(num_slices)]

            
            if self.hparams.layers > 1:                
                uv_id = clean_fname.split('_id_')[-1]
                PSF_name = f'uv_id_{uv_id}{self.hparams.PSF_file_ext}.fits'
                PSF_pth = os.path.join(PSF_path, PSF_name)
                im = fits.getdata(PSF_pth)
                im = np.expand_dims(im, 0)
                num_slices = im.shape[0]
                self.PSF += [(PSF_pth, slice) for slice in range(num_slices)]
                
            self.mat_files += [os.path.join(mat_path, f'{clean_fname}{self.hparams.mat_file_ext}.mat')]
                
        print("rec path:", root3)
        print('Number of res images:', len(self.res))
        print('Number of clean images:', len(self.clean))

    def __len__(self):
        return len(self.res)

    def __getitem__(self, i):
        fname, slice = self.clean[i]
        fname_res, slice_res = self.res[i]
        fname_rec, slice_rec = self.rec[i]

        original_image = np.expand_dims(fits.getdata(fname), axis=0)  # Adding a channel dimension. shape of output is [1, W, H]
        res = np.expand_dims(fits.getdata(fname_res), axis=0)  # Adding a channel dimension. shape of output is [1, W, H]
        rec = np.expand_dims(fits.getdata(fname_rec), axis=0)
        
        if self.hparams.layers > 1:
            fname_PSF, slice_PSF = self.PSF[i]
            PSF = fits.getdata(fname_PSF)
            PSF = np.expand_dims(PSF/PSF.max(), axis=0)
        else:
            PSF = 0.
        try:
            mat_file = loadmat(self.mat_files[i])
            try:    
                a_expo = mat_file['a_expo'].squeeze()
            except:
                try:
                    a_expo = mat_file['expo_factor'].squeeze()
                except:
                    a_expo = 0.
        except:
            a_expo = 0.
        return self.transform(original_image, res, rec, PSF, fname_res.split('/')[-1], slice, a_expo)
