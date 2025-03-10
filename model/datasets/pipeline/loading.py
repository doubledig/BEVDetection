from typing import Optional, Union, Dict, Tuple, List

import cv2
import numpy as np
import pyarrow.feather
from mmcv import BaseTransform
from mmengine.fileio import get
from numpy import random, ndarray

from model.utils import image


class LoadNusImageFromFiles(BaseTransform):
    def __init__(self,
                 color_type: str = 'unchanged',
                 scale: float = 0.5,
                 if_random: bool = False,
                 if_normalize: bool = True,
                 mean=None,
                 std=None,
                 to_rgb: bool = True,
                 pad_divisor: int = 32):
        self.color_type = color_type
        # 缩放 填充
        self._init_scale_pad(scale, pad_divisor)
        # 随机
        self.if_random = if_random
        # 归一
        self.if_normalize = if_normalize
        if mean is None:
            self.mean = np.array([(123.675, 116.28, 103.53)], dtype=np.float64)
        else:
            self.mean = np.array([mean], dtype=np.float64)
        if std is None:
            self.std = np.array([(58.395, 57.12, 57.375)], dtype=np.float64)
        else:
            self.std = np.array([std], dtype=np.float64)
        self.stdinv = 1 / np.float64(self.std)
        self.to_rgb = to_rgb

    def _init_scale_pad(self, scale, pad_divisor):
        self.if_scale = scale != 1
        # nus image 900 * 1600
        self.scale_w = max(1, int(scale * 1600))
        self.scale_h = max(1, int(scale * 900))
        self.scale = self.scale_h / 900
        # pad
        self.pad_h = int(np.ceil(self.scale_h / pad_divisor)) * pad_divisor
        self.pad_w = int(np.ceil(self.scale_w / pad_divisor)) * pad_divisor

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        images = []
        img_shapes = []
        intrinsics = []
        cam2egos = []

        # 缩放--随机--归一--填充
        for cam_type, cam_info in results['cams'].items():
            # 读取
            img = image.imread(cam_info['data_path'], flag=self.color_type)
            intrinsic = cam_info['cam_intrinsic']
            cam2egos.append(cam_info['cam2ego'])
            img = img.astype(np.float32)
            # 缩放
            if self.if_scale:
                img = cv2.resize(img, (self.scale_w, self.scale_h), interpolation=cv2.INTER_LINEAR)
                intrinsic[:2] = intrinsic[:2] * self.scale
            img_shapes.append((self.scale_h, self.scale_w))
            intrinsics.append(intrinsic)
            # 随机
            if self.if_random:
                img = self.random_img(img)
            # 归一
            if self.if_normalize:
                if self.to_rgb:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.subtract(img, self.mean)
                img = cv2.multiply(img, self.stdinv)
            # 填充
            img = self.pad_img(img)
            images.append(img)

        results['images'] = images
        results['img_shape'] = img_shapes
        results['img_padshape'] = (self.pad_h, self.pad_w)
        results['intrinsic'] = np.array(intrinsics)
        results['cam2ego'] = np.array(cam2egos)
        del results['cams']
        return results

    def pad_img(self, img: ndarray):
        width: int = max(self.pad_w - img.shape[1], 0)
        height: int = max(self.pad_h - img.shape[0], 0)
        img = cv2.copyMakeBorder(img, 0, height, 0, width, cv2.BORDER_CONSTANT, value=0)
        return img

    def random_img(self, img):
        mode = random.randint(2),
        brightness_flag = random.randint(2),
        contrast_flag = random.randint(2),
        saturation_flag = random.randint(2),
        hue_flag = random.randint(2),
        swap_flag = random.randint(2),
        delta_value = random.uniform(-32, 32),
        alpha_value = random.uniform(0.5, 1.5),
        saturation_value = random.uniform(0.5, 1.5),
        hue_value = random.uniform(-18, 18),
        swap_value = random.permutation(3)
        # random brightness
        if brightness_flag:
            img += delta_value
        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        if mode == 1:
            if contrast_flag:
                img *= alpha_value
        # convert color from BGR to HSV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # random saturation
        if saturation_flag:
            img[..., 1] *= saturation_value
        # random hue
        if hue_flag:
            img[..., 0] += hue_value
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360
        # convert color from HSV to BGR
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        # random contrast
        if mode == 0:
            if contrast_flag:
                img *= alpha_value
        # randomly swap channels
        if swap_flag:
            img = img[..., swap_value]
        return img


class LoadNusPointsToDepth(BaseTransform):
    def __init__(self,
                 grid_config: Union[tuple, list] = (1, 35),
                 down_sample: int = 1):
        self.config = grid_config
        # self.down = down_sample

    def load_point(self, path:str):
        pts_bytes = get(path, backend_args=None)
        point = np.frombuffer(pts_bytes, dtype=np.float32)
        return point.reshape(-1, 5)[:, :3]

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        # 加载点云
        point = self.load_point(results['lidar_path'])
        # 生成深度
        img_len = len(results['images'])
        depth_map_list = []
        real_size = results['img_padshape']
        lidar2ego = results['train2ego']
        for i in range(img_len):
            ego2cam = results['ego2cam'][i]
            lidar2cam = ego2cam @ lidar2ego
            intrinsic = results['intrinsic'][i]
            height, width = results['img_shape'][i]
            points = point @ lidar2cam[:3, :3].T + lidar2cam[:3, 3]
            points = points @ intrinsic.T
            points[:, :2] = points[:, :2] / points[:, [2]]
            depth = points[:, 2]
            point_id = np.round(points[:, :2]).astype(np.int32)
            depth_map = np.zeros(real_size, dtype=np.float32)
            kept1 = ((point_id[:, 0] >= 0)
                     & (point_id[:, 0] < width)
                     & (point_id[:, 1] >= 0)
                     & (point_id[:, 1] < height)
                     & (depth >= self.config[0])
                     & (depth < self.config[1]))
            point_id, depth = point_id[kept1], depth[kept1]
            ranks = point_id[:, 0] + point_id[:, 1] * real_size[1]
            sort = np.argsort(ranks + depth / 100)
            point_id, depth, ranks = point_id[sort], depth[sort], ranks[sort]
            kept2 = np.ones(point_id.shape[0], dtype=np.bool_)
            # 保留最远的
            kept2[:-1] = ranks[:-1] != ranks[1:]
            # 保留最近的
            # kept2[1:] = ranks[1:] != ranks[:-1]
            point_id, depth = point_id[kept2], depth[kept2]
            depth_map[point_id[:, 1], point_id[:, 0]] = depth
            depth_map_list.append(depth_map)
        results['depth'] = np.array(depth_map_list)
        return results


class LoadAv2ImageFromFiles(LoadNusImageFromFiles):
    def __init__(self,
                 cut_img: bool = False,
                 color_type: str = 'unchanged',
                 scale: float = 0.3,
                 if_random: bool = False,
                 if_normalize: bool = True,
                 pad_divisor: int = 32):
        self.cut_img = cut_img
        super().__init__(color_type, scale, if_random, if_normalize, pad_divisor)

    def _init_scale_pad(self, scale, pad_divisor):
        self.if_scale = scale != 1
        # nus image 1550 2048
        self.scale_w = max(1, int(scale * 2048))
        self.pad_w = int(np.ceil(self.scale_w / pad_divisor)) * pad_divisor
        self.scale_h = max(1, int(scale * 1550))
        self.pad_h = int(np.ceil(self.scale_h / pad_divisor)) * pad_divisor
        self.scale = self.scale_w / 2048

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        images = []
        img_shapes = []
        intrinsics = []
        ego2cams = []

        # 缩放--随机--归一--填充
        for cam_type, cam_info in results['cams'].items():
            # 读取
            img = image.imread(cam_info['data_path'], flag=self.color_type)
            intrinsic = cam_info['cam_intrinsic']
            ego2cams.append(cam_info['ego2cam'])
            img = img.astype(np.float32)
            # 缩放
            if self.if_scale:
                if cam_type == 'ring_front_center':
                    if self.cut_img:
                        img = img[-1550:]
                        intrinsic[1, 2] = intrinsic[1, 2] - 498
                        img = cv2.resize(img, (self.scale_h, self.scale_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        img = cv2.resize(img, (self.scale_h, self.scale_w), interpolation=cv2.INTER_LINEAR)
                else:
                    img = cv2.resize(img, (self.scale_w, self.scale_h), interpolation=cv2.INTER_LINEAR)
                intrinsic[:2] = intrinsic[:2] * self.scale
            img_shapes.append(img.shape[:2])
            intrinsics.append(intrinsic)
            # 随机
            if self.if_random:
                img = self.random_img(img)
            # 归一
            if self.if_normalize:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.subtract(img, self.mean)
                img = cv2.multiply(img, self.stdinv)
            # 填充
            img = self.pad_img(img)
            images.append(img)

        results['images'] = images
        results['img_shape'] = img_shapes
        results['img_padshape'] = (self.pad_h, self.pad_w) if self.cut_img else (self.pad_w, self.pad_w)
        results['intrinsic'] = np.array(intrinsics)
        results['ego2cam'] = np.array(ego2cams)
        del results['cams']
        return results

    def pad_img(self, img: ndarray):
        width: int = max(self.pad_w - img.shape[1], 0)
        if self.cut_img:
            height: int = max(self.pad_h - img.shape[0], 0)
        else:
            height: int = max(self.pad_w - img.shape[0], 0)
        img = cv2.copyMakeBorder(img, 0, height, 0, width, cv2.BORDER_CONSTANT, value=0)
        return img


class LoadAv2PointsToDepth(LoadNusPointsToDepth):
    def load_point(self, path:str):
        point = pyarrow.feather.read_feather(path)
        return point.values.astype(np.float32)[:, :3]

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        # 加载点云
        point = self.load_point(results['lidar_path'])
        # 生成深度
        img_len = len(results['images'])
        depth_map_list = []
        real_size = results['img_padshape']
        for i in range(img_len):
            lidar2cam = results['ego2cam'][i]
            intrinsic = results['intrinsic'][i]
            height, width = results['img_shape'][i]
            points = point @ lidar2cam[:3, :3].T + lidar2cam[:3, 3]
            points = points @ intrinsic.T
            points[:, :2] = points[:, :2] / points[:, [2]]
            depth = points[:, 2]
            point_id = np.round(points[:, :2]).astype(np.int32)
            depth_map = np.zeros(real_size, dtype=np.float32)
            kept1 = ((point_id[:, 0] >= 0)
                     & (point_id[:, 0] < width)
                     & (point_id[:, 1] >= 0)
                     & (point_id[:, 1] < height)
                     & (depth >= self.config[0])
                     & (depth < self.config[1]))
            point_id, depth = point_id[kept1], depth[kept1]
            ranks = point_id[:, 0] + point_id[:, 1] * real_size[1]
            sort = np.argsort(ranks + depth / 100)
            point_id, depth, ranks = point_id[sort], depth[sort], ranks[sort]
            kept2 = np.ones(point_id.shape[0], dtype=np.bool_)
            # 保留最远的
            kept2[:-1] = ranks[:-1] != ranks[1:]
            # 保留最近的
            # kept2[1:] = ranks[1:] != ranks[:-1]
            point_id, depth = point_id[kept2], depth[kept2]
            depth_map[point_id[:, 1], point_id[:, 0]] = depth
            depth_map_list.append(depth_map)
        results['depth'] = np.array(depth_map_list)
        return results