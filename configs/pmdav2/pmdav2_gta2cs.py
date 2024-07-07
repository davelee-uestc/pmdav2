# Default HRDA Configuration for GTA->Cityscapes
_base_ = [
    '../_base_/default_runtime.py',
    #Network Architecture
    '../_base_/models/pmdav2_r101-d8.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/cityscapes_512x512.py',
]
