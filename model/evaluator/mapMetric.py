from functools import partial
from multiprocessing import Pool
from typing import Any, Sequence, Optional

import mmengine
import numpy as np
from mmengine import print_log
from mmengine.evaluator import BaseMetric
from shapely.geometry import LineString


class MapMetric(BaseMetric):
    def __init__(self,
                 metric: str = 'chamfer',
                 classes: Sequence = None,
                 score_thresh: float = 0.0,
                 save_path: str = 'metric_data.pkl',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.classes = classes
        self.score_thresh = score_thresh
        self.metric = metric
        self.save_path = save_path
        self.num_sample = 101
        if metric == 'chamfer':
            self.thresholds = (-1.5, -1.0, -0.5)
            self.line_width = 2
        elif metric == 'chamfer_hard':
            self.thresholds = (-1.0, -0.5, -0.2)
            self.line_width = 2
        elif metric == 'iou':
            self.thresholds = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
            self.line_width = 1
        else:
            raise NotImplementedError

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        # 空间换时间，计算出的结果先储存最后使用多线程加速计算
        pred = data_samples[0]
        data_batch = data_batch['data_samples'][0]
        gt_label = data_batch.gt_label.numpy()
        mask = pred['scores'] > self.score_thresh
        result = dict()
        for i in range(len(self.classes)):
            datas = []
            mask_i = gt_label == i
            datas.append(data_batch.gt_test[mask_i])
            mask_i = pred['labels'][mask] == i
            pts = pred['pts'][mask][mask_i].numpy()
            if pts.shape[0] > 0:
                up_pts = []
                for pt in pts:
                    up_pts.append(LineString(pt))
                pts = np.array(up_pts)
            datas.append(pts)
            datas.append(pred['scores'][mask][mask_i].numpy())
            result[i] = datas
        # self.results.append(copy.deepcopy(result))
        self.results.append(result)

    def compute_metrics(self, results: list, save: bool = True) -> dict:
        if save:
            mmengine.dump(results, self.save_path)
            print_log('save temp results success!', 'current')
        print_log(f'-*-*-*-*-*-*-*-*-*-*-use metric:{self.metric}-*-*-*-*-*-*-*-*-*-*-', 'current')
        mAP = 0
        timer = mmengine.Timer()
        th_len = len(self.thresholds)
        for i in range(len(self.classes)):
            fn = partial(map_tp_fp_gen,
                         num_samples=self.num_sample,
                         line_width=self.line_width,
                         thr=self.thresholds,
                         cls=i)
            timer.since_last_check()
            pool = Pool(64)
