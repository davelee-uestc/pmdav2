## PMDAv2: Multi-Scale Prototype Matching for Domain Adaptive Semantic Segmentation

## Overview

Training deep learning models for semantic segmentation typically requires extensive labeled data, which is often scarce due to the high cost and difficulty of obtaining human annotations. However, seemingly feasible solutions such as naively applying models trained on synthetic domains to real-world domains lead to performance degradation due to the domain gaps. Domain-adaptive semantic segmentation addresses this issue by adapting models trained on labeled source domains to perform well on unlabeled target domains. However, existing methods often compromised by the conflicting between domain alignment objective and the classification objective as well as misalignment of similar classes sharing local appearances. To address this, we propose PMDAv2, a novel domain adaptation method that leverages domain-shared prototypes for domain alignment. Specifically, PMDAv2 unifies domain alignment and segmentation objectives into a single optimization framework, avoiding optimization conflicting of multiple loss terms and leading to better-aligned domains and improved overall performance. By leveraging confidence-weighted prototypes aggregating multi-scale context information, PMDAv2 effectively addresses the challenges of confusion of similar classes in prototype-based domain alignment. The proposed method demonstrates significant improvement in domain-adaptive semantic segmentation, bridging the domain gap between source and target domains more effectively. 


## Setup Environment

For this project, we used python 3.8.5. We recommend setting up a new virtual
environment:

```shell
python -m venv ~/venv/sscda
source ~/venv/sscda/bin/activate
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

## Setup Datasets

**Cityscapes:** Please, download leftImg8bit_trainvaltest.zip and
gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/)
and extract them to `data/cityscapes`.

**GTA:** Please, download all image and label packages from
[here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract
them to `data/gta`.

**Synthia (Optional):** Please, download SYNTHIA-RAND-CITYSCAPES from
[here](http://synthia-dataset.net/downloads/) and extract it to `data/synthia`.


## Testing & Predictions

The provided checkpoint trained on GTA→Cityscapes can be tested on the
Cityscapes validation set using:

```shell
python -m tools.test configs/pmdav2/pmdav2_gta2cs.py pmdav2_gta2cs_res101.pth --eval mIoU
```


## Checkpoints

Below, we provide checkpoints for GTA→Cityscapes.

* [GTA→Cityscapes with a ResNet101 backbone](https://drive.google.com/file/d/19YIXhQKgegtNUEscGQj3hUDzLrrbImWk/view?usp=sharing)

## Training

Coming soon

## Acknowledgements

PMDAv2 is based on the following open-source projects. We thank their
authors for making the source code publicly available.

* [HRDA](https://github.com/lhoyer/HRDA)
* [DAFormer](https://github.com/lhoyer/DAFormer)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)
* [SHADE](https://github.com/HeliosZhao/SHADE)

## License

This project is released under the [Apache License 2.0](LICENSE), while some
specific features in this repository are with other licenses.