from mmengine import read_base

with read_base():
    from config.base.default_runtime import *

from mmcv.cnn.bricks.transformer import FFN, BaseTransformerLayer, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention

from model.datasets.nus_map_dataset import CustomNusMapDataset
from model.datasets.pipeline.formating import MakeLineGts, PackDataToInputs
from model.datasets.pipeline.loading import LoadNusImageFromFiles
from model.detectors.maptr import MapTR
from model.backbone.resnet import ResNet
from model.neck.fpn import FPN
from model.encoder.BEVFormerTransform import BEVFormerTransform, BEVFormerLayer
from model.decoder.maptrDecoder import MapTrDecoder
from model.utils.module import TemporalAttention, GeometrySpatialCrossAttention, GeometryKernelAttention
from model.losses.loss import FocalLoss, PtsL1Loss, PtsDirCosLoss

dim = 256
bev_h = 200
bev_w = 100
map_class = ('divider', 'ped_crossing', 'boundary')
num_obj = 50
num_vec_len = 20
pc_range = (-15.0, -30.0, -2.0, 15.0, 30.0, 2.0)
real_box = (pc_range[3] - pc_range[0], pc_range[4] - pc_range[1], pc_range[5] - pc_range[2])
num_cams = 6

batch_first = True

model = dict(
    type=MapTR,
    num_obj=num_obj,
    num_cls=len(map_class),
    real_box=real_box,
    use_grid_mask=True,
    img_backbone=dict(
        type=ResNet,
        depth=50,
        frozen_stages=1,
        bn_eval=True,
        bn_frozen=True,
        pretrained='ckpts/resnet50-19c8e357.pth'  # resnet50-11ad3fa6.pth
    ),
    img_neck=dict(
        type=FPN,
        in_channels=[2048],
        out_channels=dim,
        add_extra_convs='on_output',
        num_outs=1,
        relu_before_extra_convs=True
    ),
    encoder=dict(
        type=BEVFormerTransform,
        embed_dims=dim,
        bev_h=bev_h,
        bev_w=bev_w,
        real_box=real_box,
        num_cams=num_cams,
        transformerlayer=dict(
            type=BEVFormerLayer,
            batch_first=batch_first,
            attn_cfgs=(
                dict(
                    type=TemporalAttention,
                    embed_dims=dim,
                    num_levels=1,
                    batch_first=batch_first
                ),
                dict(
                    type=GeometrySpatialCrossAttention,
                    embed_dims=dim,
                    attention=dict(
                        type=GeometryKernelAttention,
                        embed_dims=dim,
                        num_heads=4,
                        kernel_size=(3, 5),
                        num_levels=1,
                        batch_first=batch_first
                    )
                )
            ),
            ffn_cfgs=dict(
                type=FFN,
                embed_dims=dim,
                feedforward_channels=dim * 2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
        )
    ),
    decoder=dict(
        type=MapTrDecoder,
        embed_dims=dim,
        num_vec=num_obj,
        num_vec_len=num_vec_len,
        bev_h=bev_h,
        bev_w=bev_w,
        num_cls=len(map_class),
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
                    type=MultiScaleDeformableAttention,
                    embed_dims=dim,
                    num_levels=1,
                    batch_first=batch_first
                )
            ),
            ffn_cfgs=dict(
                type=FFN,
                embed_dims=dim,
                feedforward_channels=dim * 2,
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
    pts_loss=dict(
        type=PtsL1Loss,
        loss_weight=5.0
    ),
    dir_loss=dict(
        type=PtsDirCosLoss,
        loss_weight=0.005
    )
)

train_pipeline = [
    dict(type=LoadNusImageFromFiles,
         if_random=True),
    dict(type=MakeLineGts),
    dict(type=PackDataToInputs)
]

val_pipeline = [
    dict(type=LoadNusImageFromFiles,
         if_random=False),
    dict(type=MakeLineGts,
         train_mode=False),
    dict(type=PackDataToInputs)
]

train_dataloader = dict(
    dataset=dict(
        type=CustomNusMapDataset,
        ann_file='dataset/nuScenes/nus_map_train.pkl',
        data_path=data_root,
        pipeline=train_pipeline
    ),
    sampler=dict(
        type='DefaultSampler',
        shuffle=True),
    collate_fn=dict(type='default_collate'),
    batch_size=16,
    pin_memory=True,
    num_workers=4
)

val_dataloader = dict(
    dataset=dict(
        type=CustomNusMapDataset,
        ann_file='dataset/nuScenes/nus_map_val.pkl',
        data_path=data_root,
        pipeline=val_pipeline,
        test_mode=True
    ),
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),
    collate_fn=dict(type='default_collate'),
    batch_size=1,
    pin_memory=True,
    num_workers=2
)
test_dataloader = val_dataloader

train_cfg = dict(
    by_epoch=True,
    max_epochs=24,
    val_begin=25,
    val_interval=1
)

optim_wrapper = dict(
    type='AmpOptimWrapper', #
    optimizer=dict(
        type='AdamW',
        lr=3e-4,
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
        type='LinearLR', start_factor=0.3, by_epoch=False, end=1000),
    dict(
        type='CosineAnnealingLR',
        T_max=train_cfg['max_epochs'],
        by_epoch=True,
        convert_to_iter_based=True,
        eta_min_ratio=1e-3
    )
]

load_from = None
resume = False