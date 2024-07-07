# Default HRDA Configuration for GTA->Cityscapes
_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/deeplabv2_r101-d8.py',
    # GTA->Cityscapes High-Resolution Data Loading
    '../_base_/datasets/uda_gtaHR_to_cityscapesHR_1024x1024.py',
    # DAFormer Self-Training
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
name_uda = 'HRDA_a999_fdthings'
name_dataset = 'gtaHR2csHR'
# HRDA Configuration
model = dict(
    type='HRDAEncoderDecoder',
    decode_head=dict(
        type='HRDAHead',
        # Use the DAFormer decoder for each scale.
        single_scale_head='DLV2Head',
        # Learn a scale attention for each class channel of the prediction.
        attention_classwise=True,
        # Set the detail loss weight $\lambda_d=0.1$.
        hr_loss_weight=0.1),
    # Use the full resolution for the detail crop and half the resolution for
    # the context crop.
    scales=[1, 0.5],
    # Use a relative crop size of 0.5 (=512/1024) for the detail crop.
    hr_crop_size=[512, 512],
    # Use LR features for the Feature Distance as in the original DAFormer.
    feature_scale=0.5,
    # Make the crop coordinates divisible by 8 (output stride = 4,
    # downscale factor = 2) to ensure alignment during fusion.
    crop_coord_divisible=8,
    # Use overlapping slide inference for detail crops for pseudo-labels.
    hr_slide_inference=True,
    # Use overlapping slide inference for fused crops during test time.
    test_cfg=dict(
        mode='slide',
        batched_slide=True,
        stride=[512, 512],
        crop_size=[1024, 1024]))
data = dict(
    train=dict(
        # Rare Class Sampling
        # min_crop_ratio=2.0 for HRDA instead of min_crop_ratio=0.5 for
        # DAFormer as HRDA is trained with twice the input resolution, which
        # means that the inputs have 4 times more pixels.
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=2.0) if rcs_enabled else None,
        # Pseudo-Label Cropping v2:
        # Generate mask of the pseudo-label margins in the data loader before
        # the image itself is cropped to ensure that the pseudo-label margins
        # are only masked out if the training crop is at the periphery of the
        # image.
        target=dict(crop_pseudo_margins=[30, 240, 30, 30]),
    ),
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
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=5)
evaluation = dict(interval=1000, metric='mIoU')
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