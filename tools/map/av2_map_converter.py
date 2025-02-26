import argparse
import os
import time
import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import mmengine
import networkx as nx
import numpy as np
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.map.map_api import ArgoverseStaticMap
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion
from shapely import ops
from shapely.geometry import LineString, Polygon, CAP_STYLE, JOIN_STYLE, box, MultiPolygon
from shapely.strtree import STRtree

CAM_NAMES = ('ring_front_center', 'ring_front_right', 'ring_front_left',
             'ring_rear_right', 'ring_rear_left', 'ring_side_right', 'ring_side_left',
             # 'stereo_front_left', 'stereo_front_right',
             )

FAIL_LOGS = (
    # official
    '75e8adad-50a6-3245-8726-5e612db3d165',
    '54bc6dbc-ebfb-3fba-b5b3-57f88b4b79ca',
    'af170aac-8465-3d7b-82c5-64147e94af7d',
    '6e106cf8-f6dd-38f6-89c8-9be7a71e7275',
    # observed
    '01bb304d-7bd8-35f8-bbef-7086b688e35e',
    '453e5558-6363-38e3-bf9b-42b5ba0a6f1d'
)

EGO2TRAIN = np.array(
    [[0, -1, 0, 0],
     [1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]]
)

TRAIN2EGO = np.array(
    [[0, 1, 0, 0],
     [-1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]]
)

MAP_CLASS = ('divider', 'ped_crossing', 'boundary')


def get_paths(lane_dict: dict):
    pts_G = nx.DiGraph()
    junction_pts_list = []
    for key, value in lane_dict.items():
        line_geom = LineString(value['polyline'].xyz)
        line_pts = np.array(line_geom.coords).round(3)
        start_pt = line_pts[0]
        end_pt = line_pts[-1]
        for i in range(line_pts.shape[0] - 1):
            pts_G.add_edge(tuple(line_pts[i]), tuple(line_pts[i + 1]))
        valid_incoming_num = 0
        for pred in value['predecessors']:
            if pred in lane_dict.keys():
                valid_incoming_num += 1
                pred_geom = LineString(lane_dict[pred]['polyline'].xyz)
                pred_pt = np.array(pred_geom.coords).round(3)[-1]
                pts_G.add_edge(tuple(pred_pt), tuple(start_pt))
        if valid_incoming_num > 1:
            junction_pts_list.append(tuple(start_pt))
        valid_outgoing_num = 0
        for succ in value['successors']:
            if succ in lane_dict.keys():
                valid_outgoing_num += 1
                pred_geom = LineString(lane_dict[succ]['polyline'].xyz)
                pred_pt = np.array(pred_geom.coords).round(3)[0]
                pts_G.add_edge(tuple(end_pt), tuple(pred_pt))
        if valid_outgoing_num > 1:
            junction_pts_list.append(tuple(end_pt))
    roots = (v for v, d in pts_G.in_degree() if d == 0)
    leaves = [v for v, d in pts_G.out_degree() if d == 0]
    all_paths = []
    for root in roots:
        paths = nx.all_simple_paths(pts_G, root, leaves)
        all_paths.extend(paths)
    final_centerline_paths = []
    for path in all_paths:
        merged_line = LineString(path)
        final_centerline_paths.append(merged_line)
    return final_centerline_paths


def extract_local_map(avm, ego2global, pc_range):
    global2train = EGO2TRAIN @ np.linalg.inv(ego2global)
    patch_box = box(pc_range[0] - 0.1, pc_range[1] - 0.1, pc_range[3] + 0.1, pc_range[4] + 0.1)
    real_box = box(pc_range[0], pc_range[1], pc_range[3], pc_range[4])

    gts = []

    # divider
    scene_ls_list = avm.get_scenario_lane_segments()
    left_lane_dict = {}
    right_lane_dict = {}
    for ls in scene_ls_list:
        if ls.is_intersection:
            continue
        polygon = Polygon(ls.polygon_boundary @ global2train[:3, :3].T + global2train[:3, 3])
        if polygon.is_valid:
            polygon = polygon.intersection(patch_box)
            if not polygon.is_empty:
                if ls.left_lane_boundary is not None:
                    left_lane_dict[ls.id] = dict(
                        polyline=ls.left_lane_boundary,
                        predecessors=ls.predecessors,
                        successors=ls.successors,
                        neighbor_id=ls.left_neighbor_id,
                    )
                if ls.right_lane_boundary is not None:
                    right_lane_dict[ls.id] = dict(
                        polyline=ls.right_lane_boundary,
                        predecessors=ls.predecessors,
                        successors=ls.successors,
                        neighbor_id=ls.right_neighbor_id,
                    )
    for key, value in left_lane_dict.items():
        if value['neighbor_id'] in right_lane_dict.keys():
            del right_lane_dict[value['neighbor_id']]
    for key, value in right_lane_dict.items():
        if value['neighbor_id'] in left_lane_dict.keys():
            del left_lane_dict[value['neighbor_id']]
    left_paths = get_paths(left_lane_dict)
    right_paths = get_paths(right_lane_dict)
    divider_ls = left_paths + right_paths
    line_list = []
    for line in divider_ls:
        if not line.is_empty:
            line = np.array(line.coords) @ global2train[:3, :3].T + global2train[:3, 3]
            line = LineString(line)
            n_line = line.intersection(real_box)
            if not n_line.is_empty:
                if n_line.geom_type == 'MultiLineString':
                    for single_line in n_line.geoms:
                        if single_line.length >= 0.15:
                            line_list.append(single_line)
                else:
                    if n_line.length >= 0.15:
                        line_list.append(n_line)
    poly_lines = [line.buffer(1,
                              cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre) for line in line_list]
    tree = STRtree(poly_lines)
    remain_idx = [i for i in range(len(line_list))]
    for i, pline in enumerate(poly_lines):
        if i not in remain_idx:
            continue
        remain_idx.pop(remain_idx.index(i))
        gts.append(dict(
            cls_name='divider',
            type=0,
            pts=line_list[i],
        ))
        for o in tree.query(pline):
            if o not in remain_idx:
                continue
            o_line = poly_lines[o]
            inter = o_line.intersection(pline).area
            union = o_line.union(pline).area
            iou = inter / union
            if iou >= 0.90:
                remain_idx.pop(remain_idx.index(o))

    # ped_crossing
    ped_list = avm.get_scenario_ped_crossings()
    polygon_list = []
    for pc in ped_list:
        pc = pc.polygon
        pc = pc @ global2train[:3, :3].T + global2train[:3, 3]
        pc = Polygon(pc)
        if pc.is_valid:
            new_polygon = pc.intersection(real_box)
            if new_polygon.area > 0.03:
                if new_polygon.geom_type == 'Polygon':
                    new_polygon = MultiPolygon([new_polygon])
                if new_polygon.is_valid:
                    polygon_list.extend(new_polygon.geoms)
    # 利用STRtree实现合并
    # 真值
    for pc in polygon_list:
        gts.append(dict(
            cls_name='ped_crossing',
            type=1,
            pts=pc,
        ))

    # 'boundary'
    bou_list = avm.get_scenario_vector_drivable_areas()
    polygon_list = []
    for bou in bou_list:
        bou = bou.xyz @ global2train[:3, :3].T + global2train[:3, 3]
        bou = Polygon(bou)
        if bou.is_valid:
            new_polygon = bou.intersection(patch_box)
            if not new_polygon.is_empty:
                if new_polygon.geom_type == 'Polygon':
                    new_polygon = MultiPolygon([new_polygon])
                if new_polygon.is_valid:
                    polygon_list.append(new_polygon)
    geom = ops.unary_union(polygon_list)
    real_lines = []
    if geom.geom_type != 'MultiPolygon':
        geom = MultiPolygon([geom])
    for poly in geom.geoms:
        # exterior
        lines = poly.exterior.intersection(real_box)
        if lines.geom_type == 'MultiLineString':
            lines = ops.linemerge(lines)
        real_lines.append(lines)
        for inter in poly.interiors:
            lines = inter.intersection(real_box)
            if lines.geom_type == 'MultiLineString':
                lines = ops.linemerge(lines)
            real_lines.append(lines)
    for line in real_lines:
        if line.geom_type == 'MultiLineString':
            for l in line.geoms:
                if l.length >= 0.15:
                    gts.append(dict(
                        cls_name='boundary',
                        type=2,
                        pts=l,
                    ))
        elif line.length >= 0.15:
            gts.append(dict(
                cls_name='boundary',
                type=2,
                pts=line,
            ))
    return gts


def get_data_from_logid(log_id: str,
                        loader: AV2SensorDataLoader,
                        data_root: str,
                        pc_range):
    samples = []
    discarded = 0

    log_map_dirpath = Path(os.path.join(data_root, log_id, "map"))
    vector_data_fnames = sorted(log_map_dirpath.glob("log_map_archive_*.json"))
    vector_data_json_path = vector_data_fnames[0]
    avm = ArgoverseStaticMap.from_json(vector_data_json_path)
    cam_timestamps = loader._sdb.per_log_lidar_timestamps_index[log_id]

    for ts in cam_timestamps:
        cam_ring_fpath = [loader.get_closest_img_fpath(
            log_id, cam_name, ts
        ) for cam_name in CAM_NAMES]
        lidar_fpath = loader.get_closest_lidar_fpath(log_id, ts)
        if None in cam_ring_fpath or lidar_fpath is None:
            discarded += 1
            continue
        city_SE3_ego = loader.get_city_SE3_ego(log_id, int(ts))
        cams = {}
        for i, cam_name in enumerate(CAM_NAMES):
            pinhole_cam = loader.get_log_pinhole_camera(log_id, cam_name)
            cams[cam_name] = dict(
                data_path='/'.join(cam_ring_fpath[i].parts[-6:]),
                cam_intrinsic=pinhole_cam.intrinsics.K,
                ego2cam=pinhole_cam.extrinsics,
            )
        # can_bus
        e2g = city_SE3_ego.transform_matrix
        can_bus = np.ones(18)
        can_bus[:3] = e2g[:3, 3]
        rotation = Quaternion(matrix=e2g[:3, :3])
        can_bus[3:7] = rotation.elements
        patch_angle = quaternion_yaw(rotation)
        if patch_angle < 0:
            patch_angle += 2 * np.pi
        can_bus[-2] = patch_angle
        can_bus[-1] = patch_angle / np.pi * 180
        infos = dict(
            ego2global=e2g,
            cams=cams,
            lidar_path='/'.join(lidar_fpath.parts[-5:]),
            timestamp=str(ts),
            log_id=log_id,
            token=str(log_id + '_' + str(ts)),
            train2ego=TRAIN2EGO,
            can_bus=can_bus,
            # 时序待加入
        )
        # 读取gt
        map_anno = extract_local_map(avm, e2g, pc_range)
        infos['gts'] = map_anno
        samples.append(infos)
    return samples, discarded


def create_av2_infos_map(root_path: str,
                         split: str,
                         info_prefix: str,
                         save_path: str,
                         pc_range):
    root_path = os.path.join(root_path, split)
    if save_path is None:
        save_path = root_path

    loader = AV2SensorDataLoader(Path(root_path), Path(root_path))
    log_ids = list(loader.get_log_ids())
    for l in FAIL_LOGS:
        if l in log_ids:
            log_ids.remove(l)
    print(f'collecting {split} samples...')
    print('using 64 threads')
    threads = 64  # 64
    start_time = time.time()
    pool = Pool(threads)
    fn = partial(get_data_from_logid, loader=loader, data_root=root_path, pc_range=pc_range)
    rt = pool.map_async(fn, log_ids)
    pool.close()
    pool.join()
    results = rt.get()
    discarded = 0
    sample_idx = 0
    infos = []
    for _samples, _discarded in results:
        for i in range(len(_samples)):
            _samples[i]['sample_idx'] = sample_idx
            sample_idx += 1
        infos.extend(_samples)
        discarded += _discarded
    print(f'{len(infos)} available samples, {discarded} samples discarded')
    print('collected in {:.3f}s'.format(time.time() - start_time))
    metadata = dict(
        version=split,
        map_class=MAP_CLASS
    )
    data = dict(data_list=infos, metainfo=metadata)
    save_path = os.path.join(save_path, f'{info_prefix}_map_{split}.pkl')
    mmengine.dump(data, save_path)


parser = argparse.ArgumentParser(description='Av2 arg parser')
parser.add_argument('--root-path', type=str, default='dataset/av2/',
                    help='the av2 root path')
parser.add_argument('--save-path', type=str, default='dataset/av2/')
args = parser.parse_args()

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    for name in ('train', 'val', 'test'):
        create_av2_infos_map(
            root_path=args.root_path,
            split=name,
            info_prefix='av2',
            save_path=args.save_path,
            pc_range=(-14.75, -29.5, -5.0, 14.75, 29.5, 3.0)
        )
    print('finish')
