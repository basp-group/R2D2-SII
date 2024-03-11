import os
import argparse
import pathlib

# Inference related argparse functions

def prep_args_inference(args):
    """Read and process all required and optional arguments from command line.

    Parameters
    ----------
    args : _ArgumentParser
        Arguments parsed from command line.

    Returns
    -------
    _ArgumentParser
        Arguments parsed from command line and processed.
    """
    assert args.data_file.endswith('.mat') or args.data_file.endswith('.fits'), 'The provided data_file format is not currently supported. (only .fits or .mat are supported)'
    args.fname = args.data_file.split('/')[-1].split('.mat')[0].split('.fits')[0]
    args.total_num_iter = args.num_iter
    args.save_output = args.save_all_outputs
    args.mode = 'test_single'
    args.compute_metrics = False if args.gdth_file is None else True
    args.resume = False
    args.positivity = True
    if args.series == 'R2D2':
        args.layers = 1
    elif args.series == 'R3D3':
        assert args.layers > 1, 'R3D3 series must have more than 1 layer.'
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.gdth_file is None:
        print('INFO: No ground truth provided, SNR and logSNR will not be computed.')
    elif args.gdth_file is not None and args.target_dynamic_range == 0.:
        print('INFO: Ground truth provided but no target dynamic range specified, only SNR will be computed.')
    elif args.gdth_file is not None and args.target_dynamic_range > 0.:
        print('INFO: Ground truth provided and target dynamic range specified, SNR and logSNR will be computed.')
    return args

def parse_args_inference():
    """Parse all required and optional arguments from command line.

    Returns
    -------
    _ArgumentParser
        Arguments parsed from command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iter', type=int, required=True, 
                        help='Number of iterations in the R2D2/ R3D3 series, \
                            one checkpoint file required in ckpt_path for each iteration.')
    parser.add_argument('--series', choices=['R2D2', 'R3D3'], required=True,
                        help='Choose between R2D2 and R3D3 series.')
    parser.add_argument('--layers', type=int, default=1, 
                        help='Number of layers in R2D2Net for R3D3 series, default to 6.')
    parser.add_argument('--ckpt_path', type=str, required=True, 
                        help='Path to checkpoint files.')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to file containing uv coordinates, visibility data and weighting.')
    parser.add_argument('--im_dim_x', type=int, required=True,
                        help='Image dimension in x direction.')
    parser.add_argument('--im_dim_y', type=int, required=True,
                        help='Image dimension in y direction.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the output.')
    parser.add_argument('--super_resolution', type=float, required=True,
                        help='Super-resolution factor.')
    
    parser.add_argument('--operator_type', choices=['table', 'sparse_matrix'], default='table',
                        help='NUFFT for residual dirty image computation using either table interpolation or precomputed sparse matrix.')
    parser.add_argument('--gdth_file', type=str, default=None,
                        help='Path to gdth image corresponding to the dirty image, if available.')
    parser.add_argument('--target_dynamic_range', type=float, default=0.,
                        help='Dynamic range of the target image.')
    parser.add_argument('--save_all_outputs', action='store_true',
                        help='If True, reconstruction and residual dirty image in all \
                        iterations will be saved, otherwise only those from the last \
                        iteration will be saved.')
    parser.add_argument('--res_on_gpu', action='store_true',
                        help='If True, residual computation will be done on GPU.')
    parser.add_argument('--imweight_name', type=str, default='nWimag', help='Name of variable containing imweight in the uv file. Default is nWimag.')
    
    parser.add_argument('--num_chans', type=int, default=64,
                        help='Number of UNet channels')
    parser.add_argument('--num_pools', type=int, default=4,
                        help='Number of U-Net pooling layers')
    parser.add_argument('--drop_prob', type=float, default=0.,
                        help='Dropout probability')
    
    parser.add_argument('--uv_file', type=str, default=None,
                        help='.mat file containing uv data.')
    parser.add_argument('--cpus', type=int, default=1,
                        help='Number of cpus to use')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of gpus in each node')
    parser.add_argument('--nodes', type=int, default=1, 
                        help='Number of gpu nodes to be used')
    parser.add_argument('--verbose', choices=[1, 0], default=1,
                        help='Verbosity level, choose between 1 and 0.')
    return parser.parse_args()

# Pytorch lightning (training/ testing) related argparse functions

def parse_args_pl():
    """Parse all required and optional arguments from command line.

    Returns
    -------
    _ArgumentParser
        Arguments parsed from command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    
    # Model specific hyperparameters
    parser.add_argument('--num_epochs', type=int, default=50, 
                        help='Number of training epochs')
    parser.add_argument('--num_pools', type=int, default=4, 
                        help='Number of U-Net pooling layers')
    parser.add_argument('--drop_prob', type=float, default=0.0, 
                        help='Dropout probability')
    parser.add_argument('--num_chans', type=int, default=64, 
                        help='Number of U-Net channels')
    parser.add_argument('--num_chans_in', type=int, default=2,
                        help='Number of input channels')
    parser.add_argument('--num_chans_out', type=int, default=1,
                        help='Number of output channels')
    parser.add_argument('--batch_size', default=1, type=int, 
                        help='Mini batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=1000, 
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--gradient_clip_val', type=float, default=1e-2)
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--positivity', action='store_true',
                        help='If True, enforce positivity constraint on the output')
    parser.add_argument('--series', choices=['R2D2', 'R3D3'], required=True,
                        help='series to train')
    parser.add_argument('--layers', type=int, default=1, 
                        help='If > 1, R2D2-Net would be used instead of U-Net in R2D2')
    parser.add_argument('--num_iter', type=int, required=True, 
                        help='Iteration number in the R2D2/ R3D3 series to train')
    
    # Resource related arguments
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of gpus in each node')
    parser.add_argument('--nodes', type=int, default=1, 
                        help='Number of gpu nodes to be used')
    parser.add_argument('--exp_dir', type=pathlib.Path, default='experiments',
                        help='Path where the checkpoint and hyparameters will be saved')
    parser.add_argument('--exp', type=str, 
                        help='Name of the experiment')
    parser.add_argument('--resume', action='store_true',
                        help='If True, resume the training from a previous model checkpoint. ')
    parser.add_argument('--checkpoint', type=pathlib.Path,
                        help='Path to pre-trained model. Use with --mode test')
    
    # Dataset related arguments
    parser.add_argument('--data_path', type=pathlib.Path, required=True,
                        help='Path to the datasets')
    parser.add_argument('--im_dim_x', type=int, default=512,
                        help='Image dimension in x direction.')
    parser.add_argument('--im_dim_y', type=int, default=512,
                        help='Image dimension in y direction.')
    parser.add_argument('--scname_train', type=str, default='trainingset_fullnumpy', 
                        help='GT training set folder name')
    parser.add_argument('--scname_val', type=str, default='validationset_fullnumpy', 
                        help='GT validation set folder name')
    parser.add_argument('--scname_test', type=str, default='testset_fullnumpy', 
                        help='GT test set folder name')
    parser.add_argument('--rec_ext', type=str, default='_recN1', 
                        help='reconstruction input folder extension')
    parser.add_argument('--rec_file_ext', type=str, default='_rec', 
                        help='reconstruction image filename ending')
    parser.add_argument('--res_ext', type=str, default='_resN1', 
                        help='residual input folder extension')
    parser.add_argument('--dirty_file_ext', type=str, default='_dirty', 
                        help='dirty image filename ending')
    parser.add_argument('--res_file_ext', type=str, required=True,
                        help='residual image filename ending, e.g. _dirty for N1, _res for Ni')
    parser.add_argument('--gdth_file_ext', type=str, default='_gdth', 
                        help='ground truth image filename ending')
    parser.add_argument('--mat_ext', type=str, default=None,
                        help='ending of the folder with .mat files containing the dynamic range of the exponentiated image (1/a_expo).')
    parser.add_argument('--mat_file_ext', type=str, default=None,
                        help='filename ending of the files in mat_ext.')
    
    # R2D2Net/ R3D3 specific arguments
    parser.add_argument('--dirty_ext', type=str, default='_dirty', 
                        help='dirty image folder extension')
    parser.add_argument('--PSF_path', type=str, 
                        help='path to PSF files for unrolled R2D2Net')
    parser.add_argument('--PSF_file_ext', type=str, default='_PSF', 
                        help='PSF filename ending')
    
    # Inference related arguments
    parser.add_argument('--rec2_ext', type=str, default='_recN1', 
                        help='reconstruction output folder extension')
    parser.add_argument('--save_output', action='store_true', 
                        help='Save output')
    return parser.parse_args()

def parse_args_data_gen():
    main_parser = argparse.ArgumentParser()

    subparsers = main_parser.add_subparsers(dest='data_type')
    dirty_parser = subparsers.add_parser('dirty', help='Generate dirty images')
    dirty_parser.add_argument('--gdth_path', type=str, help='Path to ground truth images.')
    dirty_parser.add_argument('--super_resolution', type=float, default=1., help='Super-resolution factor.')
    dirty_parser.add_argument('--sigma_range', type=float, nargs='+', default=[2e-6,1e-3], help='Standard deviation for the noise to be added to measurement.')
    dirty_parser.add_argument('--briggs', action='store_true', help='If True, briggs weighting will be applied.')
    dirty_parser.add_argument('--expo', action='store_true', help='If True, ground truth will be exponentiated.')
    dirty_parser.add_argument('--sigma0', type=float, default=0., help='1/ current dynamic range of the ground truth image.')
    dirty_parser.add_argument('--output_dirty_path', type=str, help='Path to save output dirty images.')
    dirty_parser.add_argument('--output_gdth_path', type=str, help='Path to save output exponentiated ground truth images.')
    dirty_parser.add_argument('--output_epsilon_path', type=str, help='Path to save true noise norm in a .mat file.')
    dirty_parser.add_argument('--uv_path', type=str, help='Path to uv data')
    dirty_parser.add_argument('--imweight_name', type=str, default='nWimag', help='Name of variable containing imweight in the uv file. Default is nWimag.')
    dirty_parser.add_argument('--operator_type', choices=['table', 'sparse_matrix'], default='table',
                              help='NUFFT for residual dirty image computation using either table interpolation or precomputed sparse matrix.')
    
    dirty_parser.add_argument('--on_gpu', action='store_true', help='If True, dirty images will be computed on gpu.')

    residual_parser = subparsers.add_parser('residual', help='Generate residual dirty images')
    residual_parser.add_argument('--rec_path', type=str, help='Path to reconstructed images.')
    residual_parser.add_argument('--dirty_path', type=str, help='Path to dirty images.')
    residual_parser.add_argument('--prune', action='store_true', help='If True, dataset will be pruned according to the data-fidelity based procedure.')
    residual_parser.add_argument('--epsilon_path', type=str, default=None, help='Path to .mat files containing the value of true noise norm for pruning.')
    residual_parser.add_argument('--output_res_path', type=str, help='Path to save output residual dirty images.')
    residual_parser.add_argument('--uv_path', type=str, help='Path to uv data')
    residual_parser.add_argument('--super_resolution', type=float, default=1., help='Super-resolution factor.')
    residual_parser.add_argument('--imweight_name', type=str, default='nWimag', help='Name of variable containing imweight in the uv file. Default is nWimag.')
    residual_parser.add_argument('--operator_type', choices=['table', 'sparse_matrix'], default='table',
                                 help='NUFFT for residual dirty image computation using either table interpolation or precomputed sparse matrix.')
    residual_parser.add_argument('--on_gpu', action='store_true', help='If True, dirty images will be computed on gpu.')
    
    args = main_parser.parse_args()
    return args
