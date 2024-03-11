from model.inference import forward
from utils.evaluate import snr
from data.transforms import to_log
from utils.io import get_data, remove_lightning_console_log
from utils.args import prep_args_inference as prep_args
from utils.args import parse_args_inference as parse_args
from utils.util_model import get_DNNs, load_net
from utils.util_model import create_net_inference as create_net
from utils.misc import vprint
from data import transforms as T

import timeit
from astropy.io import fits
import torch
import numpy as np
import multiprocessing as mp

if __name__ == "__main__":
    remove_lightning_console_log()
    # parser and prepare arguments for imaging
    args = prep_args(parse_args())
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.res_device = args.device if args.res_on_gpu else torch.device('cpu')
    
    # create a pool of workers for residual computation
    if not args.res_on_gpu and args.operator_type == 'table':
        pool = mp.Pool(processes=args.cpus)
    else:
        pool = None
    vprint(f'INFO: Using {args.device} for DNN inference and {args.res_device} for residual computation.', args.verbose, 1)
    results_save = 'all iterations' if args.save_all_outputs else 'final iteration'
    vprint(f'INFO: Results from {results_save} will be saved in {args.output_path}.', args.verbose, 1)

    # initialise dictionary to store timing information
    timing = {f'N{i+1}': {'DNN inference time': None, 'Residual dirty image computation time': None} for i in range(args.total_num_iter)}
    # initialise dictionary to store reconstruction and residual dirty images
    if args.save_all_outputs:
        rec_dict = {f'N{i+1}': None for i in range(args.total_num_iter)}
        res_dict = {f'N{i+1}': None for i in range(args.total_num_iter)}
    else:
        rec_dict = {f'N{args.total_num_iter}': None}
        res_dict = {f'N{args.total_num_iter}': None}
        
    # get DNN weights and create DNN without loading weights
    dnns_dict = get_DNNs(args)
    net = create_net(args, args.device)
    
    # obtain data from args.data_file and generate starting residual dirty image and measurement operator(s)
    data, mean, op, uv, imweight, op_R2D2Net, dirty_time = get_data(args)
    dirty_norm = np.linalg.norm(data['dirty'].clone().squeeze().numpy(force=True))
    
    output = torch.zeros_like(data['dirty']).to(args.device)
    output_n = torch.zeros_like(data['dirty']).to(args.device)
    res_n = data['dirty_n']
    
    ################################################################################################
    print('-' * 10)
    print(f'{args.series} algorithm starts ...')
    start_imaging = timeit.default_timer()
    for i in range(args.num_iter):
        vprint(f'Iteration {i+1} ...', args.verbose, 1)
        start_dnn = timeit.default_timer()
        # load network weights from cpu to gpu
        net = load_net(net, i+1, args, dnns_dict)
        torch.cuda.synchronize()
        with torch.no_grad():
            output = forward(args, i, net, res_n, output_n, data, mean, op_R2D2Net)
        dnn_time = timeit.default_timer() - start_dnn
        vprint(f'INFO: DNN inference & model update time: {dnn_time:.6f} sec', args.verbose, 1)
        torch.cuda.empty_cache() 
        timing[f'N{i+1}']['DNN inference time'] = dnn_time
        start_res = timeit.default_timer()
        # generate residual dirty image
        if args.res_on_gpu or args.operator_type != 'table':
            res = op.gen_res(data['dirty'].clone().to(args.res_device), output.clone().to(args.res_device))
        else:
            res = pool.apply(op.gen_res, (data['dirty'].clone().to(args.res_device), output.clone().to(args.res_device)))
        res_time = timeit.default_timer() - start_res
        vprint(f'INFO: residual dirty image computation time: {res_time:.6f} sec', args.verbose, 1)
        
        timing[f'N{i+1}']['Residual dirty image computation time'] = res_time
        if args.save_output or (i == (args.total_num_iter-1)):
            rec_dict[f'N{i+1}'] = output
            res_dict[f'N{i+1}'] = res
        # Normalization for next iteration
        output_n, mean = T.normalize_instance(output, eps=1e-110)
        res_n = T.normalize(res, mean.to(args.res_device), eps=1e-110)
        vprint('', args.verbose, 1)
    imaging_time = timeit.default_timer() - start_imaging
    ################################################################################################
    print(f'{args.series} algorithm finished.')
    print('-' * 10)
    # report timing
    model_update_time_total = sum([timing[f"N{i+1}"]["DNN inference time"] for i in range(args.num_iter)])
    model_update_time_avg = model_update_time_total / args.num_iter
    res_time_total = sum([timing[f"N{i+1}"]["Residual dirty image computation time"] for i in range(args.num_iter-1)]) + dirty_time
    res_time_avg = res_time_total / args.num_iter
    imaging_time = model_update_time_total + res_time_total
    
    vprint('** Timings:', args.verbose, 1)
    print(f'** Total imaging time: {imaging_time:.6f} sec')
    vprint(f'** Total DNN inference & model update time: {model_update_time_total:.6f} sec', args.verbose, 1)
    vprint(f'** Total residual dirty image computation time: {res_time_total:.6f} sec', args.verbose, 1)
    vprint(f'** Average DNN inference & model update time: {model_update_time_avg:.6f} sec', args.verbose, 1)
    vprint(f'** Average residual dirty image computation time: {res_time_avg:.6f} sec', args.verbose, 1)
    
    # save reconstruction and residual dirty images
    if args.save_all_outputs:
        for i in range(args.num_iter):
            fits.writeto(f'{args.output_path}/{args.fname}_rec_N{i+1}.fits', rec_dict[f'N{i+1}'].clone().squeeze().detach().cpu().numpy(), overwrite=True)
            fits.writeto(f'{args.output_path}/{args.fname}_res_N{i+1}.fits', res_dict[f'N{i+1}'].clone().squeeze().detach().cpu().numpy(), overwrite=True)
    else:
        fits.writeto(f'{args.output_path}/{args.fname}_rec_N{args.num_iter}.fits', rec_dict[f'N{args.num_iter}'].clone().squeeze().detach().cpu().numpy(), overwrite=True)
        fits.writeto(f'{args.output_path}/{args.fname}_res_N{args.num_iter}.fits', res_dict[f'N{args.num_iter}'].clone().squeeze().detach().cpu().numpy(), overwrite=True)

    print('-' * 10)
    print('** Evaluation metrics')
    # compute and report metrics
    if args.compute_metrics:
        SNR = snr(data['gdth'].squeeze().numpy(force=True), output.squeeze().numpy(force=True))
        if args.target_dynamic_range not in [None, 0.]:
            logSNR = snr(to_log(data['gdth'],args.target_dynamic_range).squeeze().numpy(force=True), 
                         to_log(output,args.target_dynamic_range).squeeze().numpy(force=True))
            print(f'** SNR: {SNR:.6f} dB, logSNR: {logSNR:.6f} dB')
        else:
            print(f'** SNR: {SNR:.6f} dB')
    sigma_res = np.linalg.norm(res.squeeze().numpy(force=True)) / dirty_norm
    res_std = np.std(res.squeeze().numpy(force=True))
    print(f'** Data fidelity: standard deviation of the residual dirty image: {res_std:.6f}')
    print(f'** Data fidelity: ||residual|| / ||dirty||: {sigma_res:.6f}')
    print('-' * 10)
    
    print('THE END.')