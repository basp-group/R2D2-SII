"""
Functions to generate visibilities (with options to compute dirty images and PSF) and residual dirty images.
"""

import os
import pathlib
import timeit

import numpy as np
import torch
from astropy.io import fits
from scipy.io import loadmat, savemat
from tqdm import tqdm

from lib.operator import Operator
from utils.args import parse_args_data_gen
from utils.exponentiation import compute_tau, expo_im, solve_expo_factor
from utils.gen_imaging_weights import gen_imaging_weights
from utils.io import read_fits_as_tensor, read_uv
from utils.misc import vprint


def gen_visibilities(args):
    """Generate and save unweighted visibilities for the given ground truth images and uv data.

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
    print(f"Generating dirty images for {args.dataset} set ...")
    gdth_paths = [i for i in list(pathlib.Path(args.gdth_path).iterdir()) if str(i).endswith("fits")]
    print(f"from {len(gdth_paths)} ground truth images ...")
    im_size = fits.getdata(gdth_paths[0]).squeeze().shape
    device = torch.device("cuda") if args.on_gpu else torch.device("cpu")

    # Create main output path and necessary subdirectories
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    output_dirty_path = os.path.join(args.output_path, f"{args.dataset}set_fullnumpy_dirty")
    if not os.path.exists(output_dirty_path) and args.save_dirty:
        os.makedirs(output_dirty_path, exist_ok=True)
        vprint(f"Created output dirty path: {output_dirty_path}", args.verbose, 1)

    output_PSF_path = os.path.join(args.output_path, f"{args.dataset}set_fullnumpy_PSF")
    if not os.path.exists(output_PSF_path) and args.save_PSF:
        os.makedirs(output_PSF_path, exist_ok=True)
        vprint(f"Created output PSF path: {output_PSF_path}", args.verbose, 1)

    output_data_path = os.path.join(args.output_path, f"{args.dataset}set_data")
    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path, exist_ok=True)
        vprint(f"Created output data path: {output_data_path}", args.verbose, 1)

    # Exponentiated ground truth images will be saved if expo is True
    if args.expo:
        output_gdth_path = os.path.join(args.output_path, f"{args.dataset}set_fullnumpy_gdth")
        if not os.path.exists(output_gdth_path):
            vprint(f"Created output ground truth path: {output_data_path}", args.verbose, 1)
            os.makedirs(output_gdth_path, exist_ok=True)

    if type(args.seed) == int:
        seed = args.seed

    if args.uv_random:
        uv_files = list(pathlib.Path(args.uv_path).iterdir())
        uv_files = [uv for uv in uv_files if uv.suffix == ".mat"]
        uv_files = np.random.permutation(uv_files)

    for idx, gdth_file in enumerate(tqdm(gdth_paths)):
        op = Operator(im_size=im_size, op_type=args.operator_type, device=device)
        gdth = read_fits_as_tensor(gdth_file).to(device)
        vprint(f"Working on {gdth_file.name} ...", args.verbose, 1)
        fname = gdth_file.name.split(".fits")[0].split("_gdth")[0]
        if args.uv_random:
            uv_file = uv_files[idx]
            uv_fname = uv_file.name.split(".mat")[0].split("uv_id_")[-1]
            fname_uv = uv_fname
            fname = f"{fname}_id_{uv_fname}"
            if not args.expo:
                gdth_new_fname = f"{fname}_gdth.fits"
                os.system(f"mv {gdth_file} {os.path.join(args.gdth_path, gdth_new_fname)}")
        else:
            fname_uv = fname.split("_id_")[1]
            uv_file = pathlib.Path(os.path.join(args.uv_path, f"uv_id_{fname_uv}.mat"))
            uv_fname = uv_file.name.split(".mat")[0].split("uv_id_")[-1]

        if args.seed == "uv":
            uv_id = int(fname_uv.split("_")[0])
            np.random.seed(uv_id)
        else:
            np.random.seed(seed + idx)

        uv_file = os.path.join(args.uv_path, f"uv_id_{fname_uv}.mat")

        if args.SR_from_filename:
            args.super_resolution = float(uv_fname.split("_SR_")[-1].split("_BR_")[0])

        data_dict = read_uv(
            uv_file_path=uv_file,
            super_resolution=args.super_resolution,
            imweight_name=args.imweight_name,
            gen_nWimag=args.gen_nWimag,
            device=device,
            multi_noise=args.multi_noise,
        )

        if args.gen_nWimag and args.briggs:
            vprint("Briggs weighting is off for nWimag computation.", args.verbose, 1)
            args.briggs = False
        if args.return_sub_op:
            tau, sigma, new_start = compute_tau(
                op, data_dict, args.sigma_range_min, args.sigma_range_max, args.briggs, args.return_sub_op, device
            )
        else:
            tau, sigma = compute_tau(
                op, data_dict, args.sigma_range_min, args.sigma_range_max, args.briggs, args.return_sub_op, device
            )
        print(f"Computed tau: {tau}, sigma: {sigma}")

        data_dict["nW"] = 1 / tau.unsqueeze(1) if tau.size(-1) > 1 else 1 / tau

        mat_dict = {"sigma": sigma.squeeze().numpy(force=True)}

        if args.gen_nWimag:
            print("Generating imaging weights ...")
            if args.weight_type == "robust":
                if 'weight_robustness' not in data_dict:
                    match args.weight_robustness:
                        case "random":
                            weight_robustness = (
                                args.weight_robustness_min
                                + (args.weight_robustness_max - args.weight_robustness_min) * np.random.uniform()
                            )
                        case "filename":
                            weight_robustness = float(fname.split("_BR_")[-1].split("_")[0])
                        case "zero":
                            weight_robustness = 0.0
                else:
                    weight_robustness = data_dict["weight_robustness"]
            else:
                weight_robustness = None
            u = data_dict["uv"][0, 1, :]
            v = data_dict["uv"][0, 0, :]
            data_dict["nWimag"] = gen_imaging_weights(
                u, v, torch.tensor([1]), im_size, args.weight_type, args.weight_gridsize, weight_robustness
            )
        mat_dict.update({"nWimag": data_dict["nWimag"].squeeze().unsqueeze(1).numpy(force=True)})

        # Compute exponentiation factor and save exponentiated ground truth images
        if args.expo:
            assert sigma < args.sigma0, "sigma should be greater than sigma0 for exponentiation."
            expo_factor = solve_expo_factor(args.sigma0, sigma.numpy(force=True))
            vprint(
                f"Ground truth exponentiated with factor: {expo_factor} to target dynamic range {1/sigma.item():.4f}",
                args.verbose,
                1,
            )
            gdth = expo_im(gdth, expo_factor)
            fits.writeto(
                os.path.join(output_gdth_path, f"{fname}_gdth.fits"),
                gdth.squeeze().numpy(force=True),
                overwrite=True,
            )

        # Generate unweighted visibilities
        op.set_uv_imweight(data_dict["uv"], torch.tensor([1]))
        y, noise_y = op.A(gdth, tau=tau, return_noise=True)

        if args.save_vis:
            # save auxiliary data related to sampling pattern, weighting and noise
            uv_data = loadmat(uv_file)
            u = uv_data["u"]
            v = uv_data["v"]
            # Normalize u v by wavelength (speed of light / frequency)
            if "unit" in uv_data:
                if uv_data["unit"].item() == "m":
                    from scipy.constants import c

                    u = u / (c / uv_data["frequency"].squeeze())
                    v = v / (c / uv_data["frequency"].squeeze())

            mat_dict.update({"u": u, 
                             "v": v, 
                             "y": y.squeeze().unsqueeze(1).numpy(force=True)}) 

        if tau.size(-1) > 1:
            tau = tau.squeeze().numpy(force=True)
        else:
            tau = tau.item()
        tau_unique, tau_index = np.unique(tau, return_index=True)
        mat_dict.update(
            {
                "tau": tau_unique, 
                "nW": 1 / tau_unique, 
                "tau_index": tau_index
            }
        )

        if args.expo:
            mat_dict.update({"expo_factor": expo_factor})

        # Compute dirty image
        nWimag = data_dict["nWimag"]
        if args.natural_weight:
            nWimag *= data_dict['nW']
        op.set_uv_imweight(data_dict["uv"], nWimag)
        if args.save_dirty:
            dirty = op.backproj_data(y * nWimag)
            fits.writeto(
                os.path.join(output_dirty_path, f"{fname}_dirty.fits"),
                dirty.squeeze().numpy(force=True),
                overwrite=True,
            )
            if args.return_sub_op:
                for i in range(len(new_start) - 1):
                    print(f'Computing dirty for time instance {i + 1}/{len(new_start) - 1}: start: {new_start[i]}, end: {new_start[i + 1]}')
                    uv = op.uv[..., new_start[i]:new_start[i + 1]]
                    nWimag_tmp = data_dict["nWimag"][..., new_start[i]:new_start[i + 1]]
                    y_tmp = y[..., new_start[i]:new_start[i + 1]] * nWimag_tmp
                    print(y_tmp)
                    op.set_uv_imweight(uv, nWimag_tmp)
                    dirty_tmp = op.backproj_data(y_tmp)
                    fits.writeto(
                        os.path.join(output_dirty_path, f"{fname}_dirty_{i+1}.fits"),
                        dirty_tmp.squeeze().numpy(force=True),
                        overwrite=True,
                    )
        
        op.set_uv_imweight(data_dict["uv"], nWimag)
        noise = op.backproj_data(noise_y * nWimag)
        true_noise_norm = np.linalg.norm(noise.squeeze().numpy(force=True)) ** 2
        mat_dict.update({"noise": noise.squeeze().numpy(force=True), "true_noise_norm": true_noise_norm})

        # Compute PSF
        if args.save_PSF:
            PSF = op.gen_PSF(oversampling=1, normalize=True)
            fits.writeto(
                os.path.join(output_PSF_path, f"{fname}_PSF.fits"),
                PSF.squeeze().numpy(force=True),
                overwrite=True,
            )
        vprint("-" * 50, args.verbose, 1)
        mat_file_path = os.path.join(output_data_path, f"{fname}.mat")
        savemat(mat_file_path, mat_dict)
        vprint(f"Data saved to {mat_file_path}", args.verbose, 1)


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
    device = torch.device("cuda") if args.on_gpu else torch.device("cpu")

    if not os.path.exists(args.output_res_path):
        os.makedirs(args.output_res_path)
    if args.prune:
        high_path = os.path.join(args.output_res_path, "high")
        low_path = os.path.join(args.output_res_path, "low")
        os.makedirs(high_path, exist_ok=True)
        os.makedirs(low_path, exist_ok=True)
    for rec_file in tqdm(rec_paths):
        op = Operator(im_size=im_size, op_type=args.operator_type, device=device)
        fname = rec_file.name.split("_rec")[0]
        if (
            os.path.exists(os.path.join(args.output_res_path, f"{fname}_res.fits"))
            or (args.prune and os.path.exists(os.path.join(high_path, f"{fname}_res.fits")))
            or (args.prune and os.path.exists(os.path.join(low_path, f"{fname}_res.fits")))
        ):
            continue
        else:
            try:
                rec = read_fits_as_tensor(rec_file).to(device)
                fname_uv = fname.split("_id")[1]
                uv_file = os.path.join(args.uv_path, f"uv_id{fname_uv}.mat")
                data_dict = read_uv(
                    uv_file_path=uv_file,
                    super_resolution=args.super_resolution,
                    imweight_name=args.imweight_name,
                    gen_nWimag=args.gen_nWimag,
                    device=device,
                    multi_noise=args.multi_noise,
                )
                nWimag = data_dict["nWimag"]
                if args.natural_weight:
                    nWimag *= data_dict['nW']
                op.set_uv_imweight(data_dict["uv"], nWimag)

                dirty_file = os.path.join(args.dirty_path, f'{fname.split("/")[-1]}_dirty.fits')
                dirty = read_fits_as_tensor(dirty_file).to(device)
                res = op.gen_res(dirty, rec).squeeze().numpy(force=True)

                if args.prune:
                    res_norm = np.linalg.norm(res.flatten())
                    assert args.epsilon_path is not None, "epsilon_path should be provided for pruning."
                    epsilon_file = loadmat(os.path.join(args.epsilon_path, f"{fname}_epsilon.mat"))
                    epsilon = epsilon_file["true_noise_norm"].item()
                    res_norm_sqr = res_norm**2
                    if epsilon < res_norm_sqr:
                        output_path = os.path.join(args.output_res_path, "high", f"{fname}_res.fits")
                    else:
                        output_path = os.path.join(args.output_res_path, "low", f"{fname}_res.fits")
                else:
                    output_path = os.path.join(args.output_res_path, f"{fname}_res.fits")
                fits.writeto(output_path, res, overwrite=True)
            except:
                print(f"Error in processing {fname}.")


if __name__ == "__main__":
    args = parse_args_data_gen()
    match args.data_type:
        case "residual":
            gen_res(args)
        case "visibilities":
            gen_visibilities(args)
