import argparse
import os
import warnings

import mmengine
import numpy as np
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils import splits
from pyquaternion import Quaternion

CAM_NAMES = (
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
)

NuSCENES_NAME_MAPPING = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck'
}


def get_data_from_sample(nus,
                         nus_map,
                         train_scenes,
                         val_scenes,
                         test=False):
    train_infos = []
    val_infos = []
    for sample in mmengine.track_iter_progress(nus.sample):
        scene = nus.get('scene', sample['scene_token'])
        sd_rec = nus.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nus.get('calibrated_sensor',
                            sd_rec['calibrated_sensor_token'])
        pose_record = nus.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nus.get_sample_data(sample['data']['LIDAR_TOP'])
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

        info = dict(
            token=sample['token'],
            prev=sample['prev'],
            next=sample['next'],
            lidar_path=lidar_path,
            cams=cams,
            ego2global=ego2global,
            train2ego=train2ego,
        )

        # 读取 3d gt
        if not test:
            # 3d box 选择使用在lidar坐标系下的坐标
            anno = []
            nums = len(sample['anns'])
            for i in range(nums):
                box_3d = dict()
                annotation = nus.get('sample_annotation', sample['anns'][i])
                box_3d['valid_flag'] = annotation['num_lidar_pts'] + annotation['num_radar_pts'] > 0
                box_3d['num_lidar_pts'] = (annotation['num_lidar_pts'])
                box_3d['num_radar_pts'] = (annotation['num_radar_pts'])
                boxx = boxes[i]
                # x, y, z, w, l, h, rotation
                # 忽略3d框在xy方向上的旋转，假设上下面始终与地面水平
                rot = boxx.orientation.yaw_pitch_roll[0] - np.pi / 2
                while rot < np.pi / 2:
                    rot += np.pi
                # rot 转换到 `[-pi/2, pi/2]`
                box_3d['gt_box'] = np.concatenate([boxx.center,
                                                   boxx.wlh,
                                                   [rot]])
                if boxx.name in NuSCENES_NAME_MAPPING:
                    box_3d['gt_name'] = NuSCENES_NAME_MAPPING[boxx.name]
                else:
                    box_3d['gt_name'] = boxx.name
                # lidar下的速度
                velocity = nus.box_velocity(annotation['token'])
                box_3d['gt_velocity'] = np.linalg.solve(train2ego[:3, :3],
                                                        np.linalg.solve(ego2global[:3, :3], velocity))
                anno.append(box_3d)
            info['gts'] = anno
        if sample['scene_token'] in train_scenes:
            train_infos.append(info)
        elif sample['scene_token'] in val_scenes:
            val_infos.append(info)
    return train_infos, val_infos


def create_nus_infos(root_path,
                     save_path,
                     version='v1.0-trainval'):
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
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
        nusc, nus_map, train_scenes, val_scenes, test
    )

    metadata = dict(
        version=version,
        map_class=('car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle',
                   'bicycle', 'pedestrian', 'traffic_cone')
    )
    data = dict(data_list=train_infos, metainfo=metadata)
    if test:
        save_path = os.path.join(save_path, 'nus_3d_test.pkl')
        print('test sample: {}'.format(len(train_infos)))
        mmengine.dump(data, save_path)
    else:
        save_path1 = os.path.join(save_path, 'nus_3d_train.pkl')
        print('train sample: {}'.format(len(train_infos)))
        mmengine.dump(data, save_path1)
        data = dict(data_list=val_infos, metainfo=metadata)
        save_path = os.path.join(save_path, 'nus_3d_val.pkl')
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
    create_nus_infos(
        root_path=args.root_path,
        save_path=args.save_path,
        version=version,
    )
