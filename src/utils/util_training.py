from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning import Trainer
from scipy.io import loadmat
import torch
import glob
import os

def create_trainer(args):
    load_version = 0 if args.resume else None
    if args.mode == 'train':
        mat_path_files = glob.glob(os.path.join(args.data_path, args.scname_val, args.mat_ext, '*.mat'))
        if len(mat_path_files) > 0 and ('a_expo' in loadmat(mat_path_files[0]) or 'expo_factor' in loadmat(mat_path_files[0])):
            monitor = "val_logSNR_epoch"
            filename = "{epoch:02d}-{val_logSNR_epoch:.2f}"
        else:
            monitor = "val_SNR_epoch"
            filename = "{epoch:02d}-{val_SNR_epoch:.2f}"
        checkpoint_callback = [ModelCheckpoint(monitor=monitor, dirpath=args.exp_dir / "checkpoints",
                                            filename=filename, save_top_k=10, mode="max")]
        logger = TensorBoardLogger(save_dir=args.exp_dir, version=load_version)
        num_epochs = args.num_epochs
        enable_progress_bar=True
    elif 'test' in args.mode:
        logger = False
        enable_progress_bar=False
        checkpoint_callback = []
        num_epochs = 0
    if torch.cuda.is_available():
        return Trainer(
            logger=logger,
            max_epochs=num_epochs,
            devices=args.gpus,
            num_nodes=args.nodes,
            strategy='ddp',
            val_check_interval=1.,
            accelerator="gpu",
            callbacks=checkpoint_callback,
            enable_progress_bar=enable_progress_bar
        )
    else:
        print('CUDA is not available. Using CPU.')
        return Trainer(
            logger=logger,
            max_epochs=num_epochs,
            val_check_interval=1.,
            accelerator="cpu",
            callbacks=checkpoint_callback,
            enable_progress_bar=enable_progress_bar
        )
