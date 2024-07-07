# Obtained from: https://github.com/lhoyer/DAFormer
_base_ = [
    '../_base_/default_runtime.py',
    #Network Architecture
    '../_base_/models/deeplabv2_r101-d8.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/uda_gta_to_cityscapes_512x512.py',
    # DACS Self-Training
    '../_base_/uda/dacs_a999_fdthings.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 1
batch_size=2
steps=40000
rcs_enabled=True
name_encoder = 'r101'
name_decoder = 'dlv2'
name_opt = 'rcs{}'.format(rcs_enabled)
name_architecture = '{}_{}'.format(name_decoder,name_encoder)
name_uda = 'dacs_a999_fdthings'
name_dataset = 'gta2cs'
data = dict(
    train=dict(
        # Rare Class Sampling
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5) if rcs_enabled else None
    ) ,
    # Use one separate thread/worker for data loading.
    workers_per_gpu=batch_size,
    samples_per_gpu=batch_size,
)
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=steps)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=500, max_keep_ckpts=5)
evaluation = dict(interval=500, metric='mIoU')

uda = dict(
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120,
)

# Meta Information for Result Analysis
name = '{}_{}_{}_{}_b{}_{}k'\
    .format(
    name_dataset,
    name_architecture,
    name_uda,
    name_opt,
    batch_size,
    steps//1000
)
exp = 'basic'