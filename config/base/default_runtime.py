default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

custom_hooks = [
    # dict(type='EmptyCacheHook'),
    dict(type='EMAHook'),
    # dict(type='SyncBuffersHook'),
]

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

randomness = dict(
    seed=0,
    # deterministic=False,
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'

data_root = 'dataset/nuScenes/'

load_from = None
resume = False
