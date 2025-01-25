import os
from typing import Optional, List, Union, Callable

from mmengine.dataset import BaseDataset


class CustomAv2MapDataset(BaseDataset):
    """
    自定义av2数据集，用于读取重新生成的标注文件
    实现地面要素检测
    不考虑时序需求
    """
    def __init__(self,
                 ann_file: Optional[str] = '',
                 data_path: Optional[str] = '',
                 pipeline: List[Union[dict, Callable]] = None,
                 test_mode: bool = False,
                 *args,
                 **kwargs
                 ):
        self.data_path = data_path
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            test_mode=test_mode,
            *args,
            **kwargs
        )

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        # 读取并转换，相对地址变为绝对地址
        for cam_type, cam_info in raw_data_info['cams'].items():
            raw_data_info['cams'][cam_type]['data_path'] = os.path.join(self.data_path, cam_info['data_path'])
        raw_data_info['lidar_path'] = os.path.join(self.data_path, raw_data_info['lidar_path'])
        return raw_data_info

