from mmengine import read_base

with read_base():
    from ..base.default_runtime import *

from mmcv.cnn.bricks.transformer import FFN, BaseTransformerLayer, MultiheadAttention
from model.datasets.pipeline.loading import LoadNusImageFromFiles
from model.datasets.pipeline.formating import PackDataToInputs, Make3dGts
from model.datasets.nus_3d_dataset import CustomNusDataset
from model.neck.fpn import CpFPN
from model.backbone.vovnet import VoVNet
from model.detectors.petr3d import Petr3D
from model.decoder.petrDecoder import PETRDecoder
from model.losses.loss import FocalLoss, FocalCost, BBox3dL1Loss, BBox3dL1Cost

dim = 256
class_3d = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone')
num_obj = 300
detect_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
position_range = (-61.2, -61.2, -10.0, 61.2, 61.2, 10.0)
batch_first = True

model = dict(
    type=Petr3D,
    num_cls=len(class_3d),
    num_obj=num_obj,
    detect_range=detect_range,
    use_grid_mask=True,
    img_backbone=dict(
        type=VoVNet,
        frozen_stages=-1,
        bn_eval=True,
        out_features=('stage4', 'stage5'),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='ckpts/vov99-fcos3d-remapped.pth'
        )
    ),
    img_neck=dict(
        type=CpFPN,
        in_channels=[768, 1024],
        out_channels=dim,
        num_outs=2
    ),
    decoder=dict(
        type=PETRDecoder,
        num_classes=len(class_3d),
        in_channels=dim,
        num_query=num_obj * 3,
        with_position=True,
        with_multiview=True,
        load_depth=True,
        position_range=position_range,
        num_layers=6,
        transformerlayers=dict(
            type=BaseTransformerLayer,
            batch_first=batch_first,
            attn_cfgs=(
                dict(
                    type=MultiheadAttention,
                    embed_dims=dim,
                    num_heads=8,
                    attn_drop=0.1,
                    dropout_layer=dict(type='Dropout', drop_prob=0.1),
                    batch_first=batch_first
                ),
                dict(
                    type=MultiheadAttention,
                    embed_dims=dim,
                    num_heads=8,
                    attn_drop=0.1,
                    dropout_layer=dict(type='Dropout', drop_prob=0.1),
                    batch_first=batch_first
                )
            ),
            ffn_cfgs=dict(
                type=FFN,
                embed_dims=dim,
                feedforward_channels=2048,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
        )
    ),
    cls_loss=dict(
        type=FocalLoss,
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=2.0
    ),
    cls_cost=dict(
        type=FocalCost,
        weight=2.0
    ),
    box_loss=dict(
        type=BBox3dL1Loss,
        weight=0.25
    ),
    box_cost=dict(
        type=BBox3dL1Cost,
        weight=0.25
    )
)

# 原始petr图像预处理包含对于图像的随机缩放，裁剪和翻转
# 还包含对于lidar坐标系的随机旋转缩放
# petr原始R50版图像最终大小为1408*512，vov有800*320和1600*640两个大小
# 复现版R50使用缩放0.7倍，填充到1120*640的图像大小，使用随机色彩的图像预处理
# 原版处理待实现

train_pipeline = [
    dict(type=LoadNusImageFromFiles,
         if_random=True),
    dict(type=Make3dGts,
         use_valid=True,
         xy_range=detect_range,
         classes=class_3d),
    dict(type=PackDataToInputs)
]
val_pipeline = [
    dict(type=LoadNusImageFromFiles),
    dict(type=Make3dGts,
         use_valid=True,
         xy_range=detect_range,
         classes=class_3d),
    dict(type=PackDataToInputs)
]

train_dataloader = dict(
    dataset=dict(
        type=CustomNusDataset,
        ann_file='dataset/nuScenes/nus_3d_train.pkl',
        data_path=data_root,
        pipeline=train_pipeline
    ),
    sampler=dict(
        type='DefaultSampler',
        shuffle=True),
    collate_fn=dict(type='default_collate'),
    batch_size=2,
    pin_memory=True,
    num_workers=4
)
#
# val_dataloader = dict(
#     dataset=dict(
#         type=CustomNusDataset,
#         ann_file='dataset/nuScenes/nus_3d_val.pkl',
#         data_path=data_root,
#         pipeline=val_pipeline,
#         test_mode=True
#     ),
#     sampler=dict(
#         type='DefaultSampler',
#         shuffle=False),
#     collate_fn=dict(type='default_collate'),
#     batch_size=1,
#     pin_memory=True,
#     num_workers=2
# )
# test_dataloader = val_dataloader

train_cfg = dict(
    by_epoch=True,
    max_epochs=24,
    val_begin=25,
    val_interval=1
)

optim_wrapper = dict(
    type='AmpOptimWrapper',  #
    optimizer=dict(
        type='AdamW',
        lr=4e-4,
        weight_decay=0.01
    ),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }
    ),
    clip_grad=dict(max_norm=35, norm_type=2)
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.3, by_epoch=False, end=500),
    dict(
        type='CosineAnnealingLR',
        T_max=train_cfg['max_epochs'],
        by_epoch=True,
        convert_to_iter_based=True,
        eta_min_ratio=1e-3
    )
]
