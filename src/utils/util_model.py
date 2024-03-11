from model.network.unet_model import UnetModel

import torch
import glob
import gc

def get_DNNs(args, device=None):
    """Check if all DNNs are available in the specified path and return a dictionary containing the paths to all DNNs.

    Parameters
    ----------
    args : _ArgumentParser
        Arguments parsed from command line and processed.

    Returns
    -------
    dict
        Dictionary containing the paths to all DNNs.

    Raises
    ------
    ValueError
        If checkpoint for any iteration is not found or if there is a conflict in checkpoint files.
    """
    if device == None:
        device = args.device
    dnns_dict = {}
    for i in range(args.num_iter):
        dnn = glob.glob(f'{args.ckpt_path}/*N{i+1}.ckpt')
        if len(dnn) == 0:
            raise ValueError(f'Checkpoint for N{i+1} not found')
        elif len(dnn) >1:
            raise ValueError(f'Checkpoint conflict for N{i+1}')
        dnns_dict[f'N{i+1}'] = torch.load(dnn[0], map_location=device)['state_dict']
        # dnns_dict[f'N{i+1}'] = torch.load(dnn[0], map_location=torch.device('cpu'))['state_dict']
    print('All DNNs found.')
    return dnns_dict

def load_net(net, N, args, dnn_dict):
    state_dict = dnn_dict[f'N{N}']#.to(args.device)
    if args.layers == 1:
        net_tmp = {f.split(f'unet.')[1]: state_dict[f] for f in state_dict if f.startswith(f'unet')}
        net.load_state_dict(net_tmp)
        del state_dict, net_tmp
    elif args.layers > 1:
        for i in range(args.layers):
            net_tmp = {f.split(f'unet{i+1}.')[1]: state_dict[f] for f in state_dict if f.startswith(f'unet{i+1}')}
            net[i].load_state_dict(net_tmp)
            del net_tmp
        del state_dict
    gc.collect()
    return net

def create_net(args, device):
    """Create the network from the specified checkpoint file.

    Parameters
    ----------
    args : _ArgumentParser
        Arguments parsed from command line and processed.
    device : torch.device
        Device to be used for the network.

    Returns
    -------
    torch.nn.Module
        Network created from the specified checkpoint file.
    """
    load_dict = False
    if args.resume or 'test' in args.mode:
        assert args.checkpoint is not None, 'Checkpoint must be provided.'
        state_dict = torch.load(args.checkpoint, map_location=device)['state_dict']
        load_dict = True
    if args.layers == 1:
        unet = UnetModel(in_chans=2,
                         out_chans=1,
                         chans=args.num_chans,
                         num_pool_layers=args.num_pools,
                         drop_prob=0.).to(device)
        if load_dict:
            net_tmp = {f.split(f'unet.')[1]: state_dict[f] for f in state_dict if f.startswith(f'unet')}
            unet.load_state_dict(net_tmp)
    elif args.layers > 1:
        unets = []
        for i in range(args.layers):
            unet = UnetModel(in_chans=2,
                            out_chans=1,
                            chans=args.num_chans,
                            num_pool_layers=args.num_pools,
                            drop_prob=0.).to(device)
            if load_dict:
                net_tmp = {f.split(f'unet{i+1}.')[1]: state_dict[f] for f in state_dict if f.startswith(f'unet{i+1}')}
                unet.load_state_dict(net_tmp)

            unets.append(unet)
        unet = unets
    if load_dict:
        del state_dict, net_tmp
        gc.collect()
    return unet

def create_net_inference(args, device):
    """Create the network from the specified checkpoint file.

    Parameters
    ----------
    args : _ArgumentParser
        Arguments parsed from command line and processed.
    device : torch.device
        Device to be used for the network.

    Returns
    -------
    torch.nn.Module
        Network created from the specified checkpoint file.
    """
    if args.layers == 1:
        unet = UnetModel(in_chans=2,
                         out_chans=1,
                         chans=args.num_chans,
                         num_pool_layers=args.num_pools,
                         drop_prob=0.).to(device)
    elif args.layers > 1:
        unets = []
        for i in range(args.layers):
            unet = UnetModel(in_chans=2,
                            out_chans=1,
                            chans=args.num_chans,
                            num_pool_layers=args.num_pools,
                            drop_prob=0.).to(device)
            unets.append(unet)
        unet = unets
    return unet

