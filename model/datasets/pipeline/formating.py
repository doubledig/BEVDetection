from typing import Dict, Optional, Union, Tuple, List

import cv2
import numpy as np
import torch
from mmcv import BaseTransform
from mmengine.structures import BaseDataElement
from scipy.spatial.distance import cdist
from shapely import affinity
from shapely.geometry import LineString, MultiLineString, box
from simplification.cutil import simplify_coords_vw, simplify_coords_idx


class PackDataToInputs(BaseTransform):
    MATE_KEYS = (
        'prev', 'next', 'token', 'img_shape', 'img_padshape',
        'intrinsic', 'train2ego', 'ego2cam', 'can_bus', 'gt_bbox_3d',
        'depth', 'gt_label', 'gt_pts', 'gt_bev', 'gt_pv', 'gt_mask', 'gt_test',
        'gt_velocity'
    )

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        packed_results = dict()
        if 'images' in results:
            image = torch.tensor(np.array(results['images']))
            image = image.permute(0, 3, 1, 2).contiguous()
            packed_results['inputs'] = image
        data_sample = BaseDataElement()
        data_metas = {}
        for key in (
            'depth', 'intrinsic', 'ego2cam', 'train2ego', 'can_bus',
            'gt_label', 'gt_pts', 'gt_bev', 'gt_pv', 'gt_mask', 'gt_bbox_3d',
            'gt_velocity'
        ):
            if key not in results:
                continue
            results[key] = torch.tensor(results[key])

        for key in self.MATE_KEYS:
            if key in results:
                data_metas[key] = results[key]
        data_sample.set_metainfo(data_metas)
        # data_sample.set_data(data_metas)
        packed_results['data_samples'] = data_sample
        return packed_results


class Make3dGts(BaseTransform):
    def __init__(self,
                 use_valid=True,
                 xy_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                 classes: Union[Tuple, List] = None):
        self.use_valid = use_valid
        self.xy_range = xy_range
        self.classes = classes

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        gts = results['gts']
        del results['gts']
        gt_label = []
        gt_bbox = []
        gt_velocity = []
        for gt in gts:
            # 过滤不可见真值
            if self.use_valid:
                flag = gt['valid_flag']
            else:
                flag = gt['num_lidar_pts'] > 0
            if not flag:
                continue
            # 过滤类别
            if gt['gt_name'] in self.classes:
                # 过滤区域外真值
                if gt['gt_box'][0] < self.xy_range[0] or gt['gt_box'][0] > self.xy_range[3]:
                    if gt['gt_box'][1] < self.xy_range[1] or gt['gt_box'][1] > self.xy_range[4]:
                        continue
                gt_velocity.append(np.isnan(gt['gt_velocity'][0]))
                gt_bbox.append(np.concatenate([gt['gt_box'], np.nan_to_num(gt['gt_velocity'][:2])]))
                gt_label.append(self.classes.index(gt['gt_name']))
            else:
                continue
        gt_label = np.array(gt_label, dtype=np.int64)
        gt_bbox = np.array(gt_bbox, dtype=np.float32)
        gt_velocity = np.array(gt_velocity, dtype=np.bool_)
        results['gt_label'] = gt_label
        results['gt_bbox_3d'] = gt_bbox
        results['gt_velocity'] = gt_velocity
        return results


class MakeLineGts(BaseTransform):
    def __init__(self,
                 vec_len=20,
                 bev=(200, 100),
                 bev_down_sample=10 / 3,
                 feat_down_sample=32,
                 train_mode=True,
                 key_point=False):
        self.vec_len = vec_len
        self.bev = bev
        self.bev_down_sample = bev_down_sample
        self.feat_down_sample = feat_down_sample
        self.train_mode = train_mode
        self.key_point = key_point
        self.final_shift_num = self.vec_len - 1
        # if self.key_point:
        #     self.final_shift_num = 6

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        cam_num = len(results['images'])
        gts = results['gts']
        del results['gts']
        train2ego = results['train2ego']
        img_shape = results['img_shape']
        gt_semantic_mask = np.zeros((1, self.bev[0], self.bev[1]), dtype=np.uint8)
        gt_pv_semantic_mask = np.zeros((cam_num, 1,
                                        results['img_padshape'][0] // self.feat_down_sample,
                                        results['img_padshape'][1] // self.feat_down_sample), dtype=np.uint8)
        gt_label = []
        gt_test = []
        gt_pts = []
        gt_mask = []
        if self.train_mode:
            real_boxes = []
            real_box = box(-16, 0.2, 16, 35)
            for shape in img_shape:
                real_boxes.append(box(0, 0, shape[1], shape[0]))

        for gt in gts:
            is_poly = False
            is_simple = True
            if gt['pts'].geom_type == 'Polygon':
                is_poly = True
                is_simple = False
                gt['pts'] = gt['pts'].exterior
            else:
                line = np.array(gt['pts'].coords)
                if np.all(line[0] == line[-1]):
                    is_poly = True
            geom = gt['pts']
            if self.train_mode:
                # bev
                geom_bev = affinity.affine_transform(geom, (self.bev_down_sample, 0,
                                                            0, self.bev_down_sample,
                                                            self.bev[1] // 2, self.bev[0] // 2))
                coords = np.array(geom_bev.coords, dtype=np.int32)[:, :2]
                cv2.polylines(gt_semantic_mask[0],
                              np.int32([coords]),
                              False,
                              color=(1,),
                              thickness=3)
                # pv
                coords = np.array(geom.coords)
                if coords.shape[1] == 2:
                    z = np.full((coords.shape[0], 1), -train2ego[2, 3])
                    coords = np.concatenate([coords, z], axis=1)
                for i in range(cam_num):
                    e2c = results['ego2cam'][i]
                    t2c = e2c @ train2ego
                    coord = coords @ t2c[:3, :3].T + t2c[:3, 3]
                    geom_pv = LineString(coord[:, [0, 2, 1]])
                    geom_pv = geom_pv.intersection(real_box)
                    intrinsic = results['intrinsic'][i]
                    if not geom_pv.is_empty:
                        if geom_pv.geom_type == 'LineString':
                            geom_pv = MultiLineString([geom_pv])
                        elif geom_pv.geom_type != 'MultiLineString':
                            continue
                        for geom_p in geom_pv.geoms:
                            coord = np.array(geom_p.coords)[:, [0, 2, 1]]
                            coord = coord @ intrinsic.T
                            geom_p = LineString(coord[:, :2] / coord[:, [2]])
                            geom_p = geom_p.intersection(real_boxes[i])
                            if not geom_p.is_empty:
                                if geom_p.geom_type == 'LineString':
                                    geom_p = MultiLineString([geom_p])
                                elif geom_p.geom_type != 'MultiLineString':
                                    continue
                                for geom_ in geom_p.geoms:
                                    coord = np.array(geom_.coords) / self.feat_down_sample
                                    cv2.polylines(gt_pv_semantic_mask[i][0],
                                                  np.int32([coord]),
                                                  False,
                                                  color=(1,),
                                                  thickness=1)
            # gt
            coords = np.array(gt['pts'].coords)[:, :2]
            if is_simple:
                coords = simplify_coords_vw(coords, 0.1)
            geom = LineString(coords)
            if self.train_mode:
                distances = np.linspace(0, geom.length, self.vec_len)
                points = np.array([geom.interpolate(d).coords for d in distances]).reshape(-1, 2)
                shift_pts = np.full((self.final_shift_num, self.vec_len, 2), -10000, dtype=np.float32)
                if self.key_point:
                    key_points = np.array(geom.coords)
                    i = 0.1
                    while key_points.shape[0] > self.vec_len:
                        key_points = simplify_coords_vw(coords, i)
                        if i > 0.2:
                            break
                        i += 0.01
                    shift_mask = np.zeros((self.final_shift_num, self.vec_len), dtype=np.bool_)
                    mask = np.ones(self.vec_len, dtype=np.bool_)
                    if is_poly:
                        if key_points.shape[0] > self.vec_len:
                            dist = cdist(key_points[1:-1], points[1:-1])
                            row_ind, ind = key_point_match(dist)
                            row_ind = row_ind + 1
                            ind = ind + 1
                            points[ind] = key_points[row_ind]
                            mask[ind] = False
                        elif key_points.shape[0] > 2:
                            dist = cdist(points[1:-1], key_points[1:-1])
                            row_ind, ind = key_point_match(dist)
                            row_ind = row_ind + 1
                            ind = ind + 1
                            points[row_ind] = key_points[ind]
                            mask[row_ind] = False
                        else:
                            continue
                        mask[0] = False
                        points = points[:-1]
                        mask = mask[:-1]
                        for i in range(self.final_shift_num):
                            s_point = np.roll(points, i, axis=0)
                            s_mask = np.roll(mask, i, axis=0)
                            s_point = np.concatenate((s_point, s_point[[0], :]), axis=0)
                            s_mask = np.concatenate((s_mask, s_mask[[0]]), axis=0)
                            shift_pts[i] = s_point
                            shift_mask[i] = s_mask
                    else:
                        if key_points.shape[0] > self.vec_len:
                            dist = cdist(key_points[1:-1], points[1:-1])
                            row_ind, ind = key_point_match(dist)
                            row_ind = row_ind + 1
                            ind = ind + 1
                            points[ind] = key_points[row_ind]
                        elif key_points.shape[0] > 2:
                            dist = cdist(points[1:-1], key_points[1:-1])
                            row_ind, ind = key_point_match(dist)
                            row_ind = row_ind + 1
                            ind = ind + 1
                            points[row_ind] = key_points[ind]
                        idx = simplify_coords_idx(points, 0.001)
                        mask[idx] = False
                        shift_pts[0] = points
                        shift_pts[1] = np.flip(points, axis=0)
                        shift_mask[0] = mask
                        shift_mask[1] = np.flip(mask, axis=0)
                    gt_mask.append(shift_mask)
                else:
                    if is_poly:
                        points = points[:-1, :]
                        for i in range(self.vec_len - 1):
                            shift_point = np.roll(points, i, axis=0)
                            shift_point = np.concatenate((shift_point, shift_point[[0], :]), axis=0)
                            shift_pts[i] = shift_point
                    else:
                        shift_pts[0] = points
                        shift_pts[1] = np.flip(points, axis=0)
                gt_pts.append(shift_pts)
            else:
                gt_test.append(geom)
            gt_label.append(gt['type'])
        gt_label = np.array(gt_label, dtype=np.int64)
        gt_test = np.array(gt_test)
        gt_pts = np.array(gt_pts)
        gt_mask = np.array(gt_mask, dtype=np.bool_)
        results['gt_label'] = gt_label
        results['gt_pts'] = gt_pts
        results['gt_mask'] = gt_mask
        results['gt_test'] = gt_test  # 保持numpy
        results['gt_bev'] = gt_semantic_mask
        results['gt_pv'] = gt_pv_semantic_mask
        return results


def key_point_match(dist):
    dist = dist.T
    a, b = dist.shape
    cost = np.zeros((a, b))
    cost[0, 0] = dist[0, 0]
    row0 = [[] for _ in range(b)]
    row0[0].append(0)
    # 初始化
    for i in range(1, b - a + 1):
        if dist[0, i] < cost[0, i-1]:
            cost[0, i] = dist[0, i]
            row0[i].append(i)
        else:
            cost[0, i] = cost[0, i-1]
            row0[i] = row0[i-1].copy()
    # 动态规划
    for i in range(1, a):
        cost[i, i] = cost[i-1, i-1] + dist[i, i]
        row1 = [[] for _ in range(b)]
        row1[i] = row0[i-1].copy()
        row1[i].append(i)
        for j in range(i+1, b-a+1+i):
            cost_l = cost[i-1, j-1] + dist[i, j]
            if cost_l < cost[i, j-1]:
                cost[i, j] = cost_l
                row1[j] = row0[j-1].copy()
                row1[j].append(j)
            else:
                cost[i, j] = cost[i, j-1]
                row1[j] = row1[j-1].copy()
        row0 = row1.copy()
    row_ind = np.array(row0[-1])
    col_ind = np.arange(a)
    return row_ind, col_ind
