from lib.operator import operator
from lib.operator_sparse import operator as operator_sparse
from data import transforms as T

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from astropy.io import fits
from scipy.io import loadmat
import logging
import timeit
import pathlib

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
        fits.writeto(str(out_dir/fname.split(res_file_ext)[0])+f'{rec_file_ext}.fits', recons[0][1].squeeze().cpu().numpy(), overwrite=True)
        
def read_fits_as_tensor(path):
    x = fits.getdata(path)
    x = torch.tensor(x.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return x

def read_uv(uv_file_path, super_resolution, imweight_name='nWimag', device=torch.device('cpu')):
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
    uv_file = loadmat(uv_file_path)
    maxProjBaseline = uv_file['maxProjBaseline'].squeeze()
    try:
        # maxProjBaseline = np.amax(np.sqrt(uv_file['u']**2 + uv_file['v']**2))
        u = uv_file['u'].squeeze() * np.pi / (super_resolution * maxProjBaseline)
        v = uv_file['v'].squeeze() * np.pi / (super_resolution * maxProjBaseline)
        uv = torch.tensor(np.stack((-v, u), axis=1).astype(np.float32)).squeeze().unsqueeze(0)
        if uv.size(1) > uv.size(2):
            uv = uv.permute(0, 2, 1)
    except:
        uv = uv_file['uv'].squeeze()
        if uv.shape[1] < uv.shape[0]:
            uv = uv.T
        maxProjBaseline_from_uv = np.amax(np.sqrt(uv[0,:]**2 + uv[1,:]**2))
        if np.isclose(maxProjBaseline, maxProjBaseline_from_uv, rtol=1e-3):
            print(f'WARNING: maxProjBaseline from data file: {maxProjBaseline} is not close to the value computed from uv of the data file, please check!')
        uv = uv * np.pi / (super_resolution * maxProjBaseline)
        uv = torch.tensor(uv.astype(np.float32)).squeeze().unsqueeze(0)
    if len(imweight_name.split(',')) > 1:
        imweight_names = imweight_name.split(',')
        try:
            imweight = torch.tensor(uv_file[imweight_names[0]].astype(np.float32).squeeze()).unsqueeze(0).unsqueeze(0)
        except:
            imweight = torch.tensor(uv_file[imweight_names[1]].astype(np.float32).squeeze()).unsqueeze(0).unsqueeze(0)
    else:
        imweight = torch.tensor(uv_file[imweight_name].astype(np.float32).squeeze()).unsqueeze(0).unsqueeze(0)
        
    uv = uv.to(device)
    imweight = imweight.to(device)
    return uv, imweight

def get_data(args):
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
    op = operator(im_size=im_size, op_type=args.operator_type, op_acc='exact', device=args.res_device)
    if args.data_file.endswith('.fits'):
        # print(f'INFO: Reading dirty image from {args.data_file}.')
        dirty = read_fits_as_tensor(args.data_file).to(args.device)
        assert args.uv_file is not None, "uv_file must be provided for .fits data."
        assert args.uv_file.endswith('.mat'), 'uv file must end format.'
        uv, imweight = read_uv(args.uv_file, args.super_resolution, args.imweight_name, device=args.res_device)
        dirty_time = 0
        op.set_uv_imweight(uv, imweight)
    elif args.data_file.endswith('.mat'):
        data = loadmat(args.data_file)
        for var in ['y', 'u', 'v', 'nWimag', 'nW']:
            assert var in data, f'{var} not found in data_file.'
        # print(f'INFO: Reading data from {args.data_file}, \ncreating corresponding dirty image.')
        uv, imweight = read_uv(args.data_file, args.super_resolution, args.imweight_name, device=args.res_device)
        if 'nW' in data:
            nW = data['nW'].squeeze()
            baselines = np.sqrt(data['u'].squeeze()**2 + data['v'].squeeze()**2)
            flag = (baselines>0) * (nW>0)
            if flag.sum() < len(flag):
                print(f'INFO: flagging {len(flag) - flag.sum()} visibilities.')
                y = y[0, 0, flag]
                uv = uv[0, :, flag]
                imweight = imweight[0, 0, flag]
        op.set_uv_imweight(uv, imweight)
        ts = timeit.default_timer()
        y = data['y'].squeeze() #* data['nW'].squeeze()
        y = y.real.astype(np.float32) + 1j * y.imag.astype(np.float32)
        y = torch.tensor(y).unsqueeze(0).unsqueeze(0).to(args.res_device) * imweight
        dirty = op.backproj_data(y).to(args.device)
        dirty_time = timeit.default_timer() - ts
        print(f'INFO: time to compute dirty image (normalized backprojected data): {dirty_time:.6f} sec')
        fits.writeto(f'{args.output_path}/{args.fname}_dirty.fits', dirty.clone().squeeze().numpy(force=True), overwrite=True)
    dirty_n, mean = T.normalize_instance(dirty, eps=1e-110)
    if args.gdth_file is not None:
        gdth = read_fits_as_tensor(args.gdth_file).to(args.device)
        gdth_n = T.normalize(gdth, mean, eps=1e-110)
    else:
        gdth = 0.
        gdth_n = 0.
    
    if args.layers > 1:
        # print(f'R2D2-Net with {args.layers} layers chosen, generating PSF.')
        uv_PSF = uv.clone().to(args.device)
        imweight_PSF = imweight.clone().to(args.device)
        os = 2
        PSF_im_size = (os*im_size[0], os*im_size[1])
        op_R2D2Net = operator(im_size=im_size, op_acc='approx', device=args.device)
        op_tmp = operator(im_size=PSF_im_size, op_acc='exact', device=args.device)
        op_tmp.set_uv_imweight(uv_PSF, imweight_PSF)
        time_PSF = timeit.default_timer()
        PSF = op_tmp.gen_PSF(normalize=True)
        fits.writeto(f'{args.output_path}/{args.fname}_PSF.fits', PSF.clone().squeeze().numpy(force=True), overwrite=True)
        print(f'INFO: time to compute the PSF needed in R2D2-Net: {timeit.default_timer() - time_PSF:.6f} sec')
        del uv_PSF, imweight_PSF
    else:
        PSF = 0.
        op_R2D2Net = None
    data = {'dirty': dirty, 
             'dirty_n': dirty_n, 
             'gdth': gdth, 
             'gdth_n': gdth_n, 
             'PSF': PSF, 
             'fname': args.fname}
    return data, mean, op, uv, imweight, op_R2D2Net, dirty_time

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
        root = f'{self.hparams.data_path}/{data_partition}' # ground truth
        root2 = f'{self.hparams.data_path}/{data_partition}{self.hparams.res_ext}' # dirty/ residual dirty
        mat_path = f'{self.hparams.data_path}/{data_partition}{self.hparams.mat_ext}' # dynamic range of the exponentiated image (1/a)
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
                clean_pth = f'{root}/{clean_fname}{self.hparams.gdth_file_ext}.fits'
            else:
                clean_pth = f'{root}/{clean_fname}.fits'
            im = fits.getdata(clean_pth)
            im = np.expand_dims(im, 0)
            num_slices = im.shape[0]
            self.clean += [(clean_pth, slice) for slice in range(num_slices)]
            
            if self.hparams.layers > 1:
                uv_id = clean_fname.split('_id_')[-1]
                PSF_name = f'uv_id_{uv_id}{self.hparams.PSF_file_ext}.fits'
                PSF_pth = f'{PSF_path}/{PSF_name}'
                im = fits.getdata(PSF_pth)
                im = np.expand_dims(im, 0)
                num_slices = im.shape[0]
                self.PSF += [(PSF_pth, slice) for slice in range(num_slices)]
                
            self.mat_files += [f'{mat_path}/{clean_fname}{self.hparams.mat_file_ext}.mat']
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
        root = self.hparams.data_path / f'{data_partition}' # ground truth
        root2 = self.hparams.data_path / f'{data_partition}{self.hparams.res_ext}' # residual dirty
        root3 = self.hparams.data_path / f'{data_partition}{self.hparams.rec_ext}' # reconstruction
        mat_path = f'{self.hparams.data_path}/{data_partition}{self.hparams.mat_ext}' # dynamic range of the exponentiated image (1/a)
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
            
            clean_pth = str(root.resolve()) + '/' + clean_fname + f'{self.hparams.gdth_file_ext}.fits'
            im = fits.getdata(clean_pth)
            im = np.expand_dims(im, 0)
            num_slices = im.shape[0]
            self.clean += [(clean_pth, slice) for slice in range(num_slices)]

            rec_pth = str(root3.resolve()) + '/' + clean_fname + f'{self.hparams.rec_file_ext}.fits'
            im = fits.getdata(rec_pth)
            im = np.expand_dims(im, 0)
            num_slices = im.shape[0]
            self.rec += [(rec_pth, slice) for slice in range(num_slices)]

            
            if self.hparams.layers > 1:                
                uv_id = clean_fname.split('_id_')[-1]
                PSF_name = f'uv_id_{uv_id}{self.hparams.PSF_file_ext}.fits'
                PSF_pth = f'{PSF_path}/{PSF_name}'
                im = fits.getdata(PSF_pth)
                im = np.expand_dims(im, 0)
                num_slices = im.shape[0]
                self.PSF += [(PSF_pth, slice) for slice in range(num_slices)]
                
            self.mat_files += [f'{mat_path}/{clean_fname}{self.hparams.mat_file_ext}.mat']
                
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