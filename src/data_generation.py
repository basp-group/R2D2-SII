import os
import argparse
import pathlib
import numpy as np
from astropy.io import fits
from scipy.io import loadmat, savemat
import torch
from tqdm import tqdm

from utils.io import read_fits_as_tensor, read_uv
from utils.args import parse_args_data_gen as parse_args
from utils.misc import solve_expo_factor
from lib.operator import operator

def gen_dirty(args):
    """Generate and save dirty images for the given ground truth images and uv data.

    Parameters
    ----------
    uv_path : str
        Path to uv data.
    gdth_path : str
        Path to the ground truth images.
    output_dirty_path : str
        Path to save the dirty images.
    output_gdth_path : str
        Path to save the exponentiated gdth images.
    super_resolution : float, optional
        Super resolution factor, by default 1.
    sigma_range : tuple, optional
        If a tuple of (min, max) is provided, sigma will be uniformly sampled from 
        U[min, max] and corresponding noise will be added to the measurement, by default None.
    briggs : bool, optional
        If True, briggs weighting will be applied, by default False.
    expo : bool, optional
        If True, ground truth will be exponentiated, by default False.
    sigma0 : float, optional
        1/ current dynamic range of the ground truth image, by default 0.
    imweight_name : str, optional
        Specific name of variable containing the weighting to be used in the uv file, by default 'nWimag'.
    on_gpu : bool, optional
        If True, dirty images will be computed on gpu, by default False.
    """
    print(f'Generating dirty images for {args.dataset} set ...')
    gdth_paths = list(pathlib.Path(args.gdth_path).iterdir())
    print(f'from {len(gdth_paths)} ground truth images ...')
    im_size = fits.getdata(gdth_paths[0]).squeeze().shape
    device = torch.device('cuda') if args.on_gpu else torch.device('cpu')
    op = operator(im_size=im_size, op_type=args.operator_type, op_acc='exact', device=device)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    args.output_dirty_path = f'{args.output_path}/{args.dataset}set_dirty'
    if not os.path.exists(args.output_dirty_path):
        os.makedirs(args.output_dirty_path)
    if args.expo:
        args.output_gdth_path = f'{args.output_path}/{args.dataset}set_gdth'
        if not os.path.exists(args.output_gdth_path):
            os.makedirs(args.output_gdth_path)
        args.output_epsilon_path = f'{args.output_path}/{args.dataset}set_epsilon'
        if not os.path.exists(args.output_epsilon_path):
            os.makedirs(args.output_epsilon_path)

    for gdth_file in tqdm(gdth_paths):
        gdth = read_fits_as_tensor(gdth_file).to(device)
        fname = gdth_file.name.split('.fits')[0].split('_gdth')[0]
        fname_uv = fname.split('_id_')[1]
        uv_file = f'{args.uv_path}/uv_id_{fname_uv}.mat'
        uv, imweight = read_uv(uv_file, args.super_resolution, args.imweight_name, device)
        op.set_uv_imweight(uv, imweight)
        if args.sigma_range is not None:
            assert len(args.sigma_range) == 2, 'sigma_range should be a tuple of (min, max).'
            op_norm = op.op_norm()
            log_sigma = (np.random.rand(1, 1) * (np.log10(args.sigma_range[1]) - np.log10(args.sigma_range[0]))
                     + np.log10(args.sigma_range[0]))
            sigma = (10 ** log_sigma)[0][0]
            sigma = torch.tensor(sigma).to(device)
            tau = sigma * torch.sqrt(2 * op_norm)
            if args.briggs:
                op_norm_double = op.op_norm()
                tau = tau / torch.sqrt(op_norm_double / op_norm)
        else:
            tau = 0
            
        if args.expo:
            assert args.sigma_range is not None, 'sigma_range should be provided for exponentiation.'
            assert args.sigma0 > 0, 'sigma0 should be greater than 0 for exponentiation.'
            assert sigma < args.sigma0, 'sigma should be greater than sigma0 for exponentiation.'
            expo_factor = solve_expo_factor(args.sigma0, sigma.numpy(force=True))
            gdth = (expo_factor**gdth - 1) / expo_factor
            fits.writeto(os.path.join(args.output_gdth_path, f'{fname}_gdth.fits'), gdth.squeeze().numpy(force=True), overwrite=True)
            savemat(os.path.join(args.output_epsilon_path, f'{fname}_epsilon.mat'),
                    {'sigma': sigma.item(),
                     'true_noise_norm': np.linalg.norm(tau.squeeze().numpy(force=True))**2})
            
        dirty = op.backproj(gdth, tau=tau)
        fits.writeto(os.path.join(args.output_dirty_path, f'{fname}_dirty.fits'), dirty.squeeze().numpy(force=True), overwrite=True)
                
def gen_res(args):
    """Compute and save residual dirty images for the given reconstructed imagesm dirty images and uv data.

    Parameters
    ----------
    uv_path : str
        Path to uv data.
    rec_path : str
        Path to reconstructed images.
    dirty_path : str
        Path to dirty images.
    output_res_path : str
        Path to save the residual dirty images.
    super_resolution : float, optional
        Super resolution factor, by default 1.
    epsilon_path : str, optional
        Path to .mat files containing true noise norm for pruning, by default None.
    prune : bool, optional
        If True, dataset will be pruned, by default False.
    imweight_name : str, optional
        Specific name of variable containing the weighting to be used in the uv file, by default 'nWimag'.
    on_gpu : bool, optional
        If True, dirty images will be computed on gpu, by default False.
    """
    rec_paths = list(pathlib.Path(args.rec_path).iterdir())
    im_size = fits.getdata(rec_paths[0]).squeeze().shape
    device = torch.device('cuda') if args.on_gpu else torch.device('cpu')
    op = operator(im_size=im_size, op_type=args.operator_type, op_acc='exact', device=device)

    if not os.path.exists(args.output_res_path):
        os.makedirs(args.output_res_path)
    if args.prune:
        high_path = os.path.join(args.output_res_path, 'high')
        low_path = os.path.join(args.output_res_path, 'low')
        os.makedirs(high_path, exist_ok=True)
        os.makedirs(low_path, exist_ok=True)
    for rec_file in tqdm(rec_paths):
        fname = rec_file.name.split('_rec')[0]
        if os.path.exists(f'{args.output_res_path}/{fname}_res.fits') or os.path.exists(f'{high_path}/{fname}_res.fits') or os.path.exists(f'{low_path}/{fname}_res.fits'):
            continue
        else:
            try:
                rec = read_fits_as_tensor(rec_file).to(device)
                fname_uv = fname.split('_id')[1]
                uv_file = f'{args.uv_path}/uv_id{fname_uv}.mat'
                uv, imweight = read_uv(uv_file, args.super_resolution, args.imweight_name, device)
                op.set_uv_imweight(uv, imweight)
                    
                dirty_file = f'{args.dirty_path}/{fname.split("/")[-1]}_dirty.fits'
                dirty = read_fits_as_tensor(dirty_file).to(device)
                res = op.gen_res(dirty, rec).squeeze().numpy(force=True)

                if args.prune:
                    res_norm = np.linalg.norm(res.flatten())
                    assert args.epsilon_path is not None, 'epsilon_path should be provided for pruning.'
                    epsilon_file = loadmat(os.path.join(args.epsilon_path, f'{fname}_epsilon.mat'))
                    epsilon = epsilon_file['true_noise_norm'][0][0]
                    res_norm_sqr = res_norm ** 2
                    if epsilon < res_norm_sqr:
                        fits.writeto(os.path.join(args.output_res_path, 'high', f'{fname}_res.fits'), res, overwrite=True)
                    else:
                        fits.writeto(os.path.join(args.output_res_path, 'low', f'{fname}_res.fits'), res, overwrite=True)
                else:
                    fits.writeto(os.path.join(args.output_res_path, f'{fname}_res.fits'), res, overwrite=True)
            except:
                print(f'Error in processing {fname}.')

if __name__ == '__main__':
    args = parse_args()
    if args.data_type == 'dirty':
        gen_dirty(args)
    elif args.data_type == 'residual':
        gen_res(args)