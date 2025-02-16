import argparse
import os
import warnings

import mmengine
from mmengine import Config, EVALUATOR
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config',
                        help='train config file path')
    parser.add_argument('load_from',
                        help='load model from a checkpoint')
    parser.add_argument('--work_dir',
                        default=None)
    parser.add_argument('--load_result',
                        default=None,
                        type=str)
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
        config.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])
    else:
        config.work_dir = args.work_dir
    if args.load_from is not None:
        config.load_from = args.load_from
    config.test_evaluator['save_path'] = os.path.join(config.work_dir, 'metric_data.pkl')

    if args.load_result is not None:
        metric = EVALUATOR.build(config.test_evaluator)
        result = mmengine.load(args.load_result)
        maps = metric.compute_metrics(result, False)
        print(maps)
    else:
        runner = Runner.from_cfg(config)
        runner.test()
