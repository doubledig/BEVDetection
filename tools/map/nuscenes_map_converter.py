import argparse
import os
import warnings

import mmengine
import numpy as np
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils import splits
from pyquaternion import Quaternion
from shapely import box, ops, MultiLineString, MultiPolygon

NUM_CLASS = dict(
    divider=('road_divider', 'lane_divider'),
    ped_crossing=('ped_crossing',),
    boundary=('road_segment', 'lane'),
    # intersection=('road_segment',),
    # stop_line=('stop_line',),
)

CAM_NAMES = (
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
)


def get_map_samples(nus_map, ego2global, train2ego):
    train2global = ego2global @ train2ego
    pose_center = tuple(train2global[:2, 3])
    rotation = Quaternion(matrix=train2global)
    patch_box = (pose_center[0], pose_center[1], 60.2, 30.2)
    patch_angle = quaternion_yaw(rotation) / np.pi * 180

    gts = []
    real_box = box(-14.75, -29.5, 14.75, 29.5)

    # divider
    geoms = nus_map.get_map_geom(patch_box, patch_angle, NUM_CLASS['divider'])
    for geom in geoms:
        for line in geom[1]:
            line = line.intersection(real_box)
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    line = ops.linemerge(line)
                if line.geom_type == 'LineString':
                    line = MultiLineString([line])
                for l in line.geoms:
                    if l.length >= 0.15:
                        gts.append(dict(
                            cls_name='divider',
                            type=0,
                            pts=l
                        ))
    # ped_crossing
    geoms = nus_map.get_map_geom(patch_box, patch_angle, NUM_CLASS['ped_crossing'])
    geom = geoms[0][1]
    for multipolygon in geom:
        for polygon in multipolygon.geoms:
            polygon = polygon.intersection(real_box)
            if polygon.area > 0.03:
                if polygon.geom_type == 'Polygon':
                    polygon = MultiPolygon([polygon])
                for l in polygon.geoms:
                    gts.append(dict(
                        cls_name='ped_crossing',
                        type=1,
                        pts=l
                    ))

    # boundary
    geoms = nus_map.get_map_geom(patch_box, patch_angle, NUM_CLASS['boundary'])
    multipolygon = geoms[0][1]
    multipolygon.extend(geoms[1][1])
    geoms = ops.unary_union(multipolygon)
    if geoms.geom_type == 'Polygon':
        geoms = MultiPolygon([geoms])
    real_lines = []
    for geom in geoms.geoms:
        lines = geom.exterior.intersection(real_box)
        if lines.geom_type == 'MultiLineString':
            lines = ops.linemerge(lines)
        real_lines.append(lines)
        for inter in geom.interiors:
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


def get_data_from_sample(nus,
                         nus_can_bus,
                         nus_map,
                         train_scenes,
                         val_scenes,
                         test=False):
    train_infos = []
    val_infos = []
    for sample in mmengine.track_iter_progress(nus.sample):
        scene = nus.get('scene', sample['scene_token'])
        map_location = nus.get('log', scene['log_token'])['location']
        sd_rec = nus.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nus.get('calibrated_sensor',
                            sd_rec['calibrated_sensor_token'])
        pose_record = nus.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path = sd_rec['filename']
        train2ego = np.eye(4)
        train2ego[:3, :3] = Quaternion(cs_record['rotation']).rotation_matrix
        train2ego[:3, 3] = cs_record['translation']
        ego2global = np.eye(4)
        rotation = Quaternion(pose_record['rotation'])
        ego2global[:3, :3] = rotation.rotation_matrix
        ego2global[:3, 3] = pose_record['translation']

        cams = {}
        for cam in CAM_NAMES:
            cam_token = sample['data'][cam]
            _, _, cam_intrinsic = nus.get_sample_data(cam_token)
            sd_cam = nus.get('sample_data', cam_token)
            cs_cam = nus.get('calibrated_sensor', sd_cam['calibrated_sensor_token'])
            cam2ego = np.eye(4)
            cam2ego[:3, :3] = Quaternion(cs_cam['rotation']).rotation_matrix
            cam2ego[:3, 3] = cs_cam['translation']
            cams[cam] = dict(
                data_path=sd_cam['filename'],
                cam_intrinsic=cam_intrinsic,
                ego2cam=np.linalg.inv(cam2ego),
            )

        # can_bus
        sample_timestamp = sample['timestamp']
        can_bus = np.zeros(18)
        can_bus[:3] = pose_record['translation']
        can_bus[3:7] = pose_record['rotation']
        patch_angle = quaternion_yaw(rotation)
        if patch_angle < 0:
            patch_angle += 2 * np.pi
        can_bus[-2] = patch_angle
        can_bus[-1] = patch_angle / np.pi * 180
        try:
            pose_list = nus_can_bus.get_messages(scene['name'], 'pose')
            last_pose = pose_list[0]
            for pose in pose_list[1:]:
                if pose['utime'] > sample_timestamp:
                    break
                last_pose = pose
            can_bus[7:10] = last_pose['accel']
            can_bus[10:13] = last_pose['rotation_rate']
            can_bus[13:16] = last_pose['vel']
        except:
            can_bus[7:16] = 0

        info = dict(
            token=sample['token'],
            prev=sample['prev'],
            next=sample['next'],
            lidar_path=lidar_path,
            map_location=map_location,
            cams=cams,
            ego2global=ego2global,
            train2ego=train2ego,
            can_bus=can_bus,
        )

        # 读取gt
        if not test:
            map_anno = get_map_samples(nus_map[map_location], ego2global, train2ego)
            info['gts'] = map_anno
        if sample['scene_token'] in train_scenes:
            train_infos.append(info)
        elif sample['scene_token'] in val_scenes:
            val_infos.append(info)
    return train_infos, val_infos


def create_nus_infos_map(root_path,
                         save_path,
                         version='v1.0-trainval'):
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    nusc_can_bus = NuScenesCanBus(dataroot=root_path)
    nus_map = {}
    for loc in ('boston-seaport', 'singapore-hollandvillage', 'singapore-onenorth', 'singapore-queenstown'):
        nus_map[loc] = NuScenesMap(dataroot=root_path, map_name=loc)

    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    # 转换为对应的token
    t, v = 0, 0
    for scene in nusc.scene:
        if scene['name'] == train_scenes[t]:
            train_scenes[t] = scene['token']
            t = t + 1
        elif scene['name'] == val_scenes[v]:
            val_scenes[v] = scene['token']
            v = v + 1

    train_scenes = set(train_scenes)
    val_scenes = set(val_scenes)

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))

    train_infos, val_infos = get_data_from_sample(
        nusc, nusc_can_bus, nus_map, train_scenes, val_scenes, test
    )

    metadata = dict(
        version=version,
        map_class=('divider', 'ped_crossing', 'boundary')
    )
    data = dict(data_list=train_infos, metainfo=metadata)
    if test:
        save_path = os.path.join(save_path, 'nus_map_test.pkl')
        print('test sample: {}'.format(len(train_infos)))
        mmengine.dump(data, save_path)
    else:
        save_path1 = os.path.join(save_path, 'nus_map_train.pkl')
        print('train sample: {}'.format(len(train_infos)))
        mmengine.dump(data, save_path1)
        data = dict(data_list=val_infos, metainfo=metadata)
        save_path = os.path.join(save_path, 'nus_map_val.pkl')
        print('train sample: {}'.format(len(val_infos)))
        mmengine.dump(data, save_path)


parser = argparse.ArgumentParser(description='NuScenes arg parser')
parser.add_argument('--root-path', type=str, default='dataset/nuScenes/',
                    help='the NuScenes root path')
parser.add_argument('--save-path', type=str, default='dataset/nuScenes/')
args = parser.parse_args()

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    version = 'v1.0-trainval'  # 'v1.0-test' 'v1.0-mini'
    create_nus_infos_map(
        root_path=args.root_path,
        save_path=args.save_path,
        version=version,
    )
