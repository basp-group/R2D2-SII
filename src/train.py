from model.R2D2_model import R2D2_model
from utils.util_training import create_trainer
from utils.args import parse_args_pl as parse_args

if __name__ == '__main__':
    args = parse_args()
    
    if args.series == 'R2D2':
        assert args.layers == 1, 'R2D2 is only defined for layers=1'
    elif args.series == 'R3D3':
        assert args.layers > 1, 'R3D3 is only defined for layers>1'
        
    model = R2D2_model(args)
        
    if args.mode == 'train':
        trainer = create_trainer(args)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('Total number of params:', pytorch_total_params)
        trainer.fit(model, ckpt_path=args.checkpoint)
    else:
        args.mode == 'test'
        assert args.checkpoint is not None
        for attr in args.__dir__():
            setattr(model.hparams, attr, getattr(args, attr))
        trainer = create_trainer(args)
        trainer.test(model)