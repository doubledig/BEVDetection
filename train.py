import argparse
import os
import warnings
from os import path as osp

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config',
                        help='train config file path')
    parser.add_argument('--load_from',
                        default=None)
    parser.add_argument('--resume',
                        action='store_true')
    parser.add_argument('--work_dir',
                        default=None)
    parser.add_argument('--launcher',
                        action='store_true')
    parser.add_argument('--gpus',
                        type=int,
                        default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = parse_args()
    config = Config.fromfile(args.config)
    if args.work_dir is None:
        config.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    else:
        config.work_dir = args.work_dir
    if args.load_from is not None:
        config.load_from = args.load_from
    if args.resume:
        config.resume = True
    if args.launcher:
        config.launcher = 'pytorch'
        config.auto_scale_lr = dict(
            enable=True,
            base_batch_size=config.train_dataloader['batch_size'] * args.gpus
        )

    runner = Runner.from_cfg(config)
    runner.train()
