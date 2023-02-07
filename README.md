[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/top-down-beats-bottom-up-in-3d-instance/3d-instance-segmentation-on-s3dis)](https://paperswithcode.com/sota/3d-instance-segmentation-on-s3dis?p=top-down-beats-bottom-up-in-3d-instance)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/top-down-beats-bottom-up-in-3d-instance/3d-instance-segmentation-on-scannetv2)](https://paperswithcode.com/sota/3d-instance-segmentation-on-scannetv2?p=top-down-beats-bottom-up-in-3d-instance)

## Top-Down Beats Bottom-Up in 3D Instance Segmentation

**News**:
 * :fire: February 6, 2023. We achieved SOTA results on the S3DIS validation subset (mAP@50) and ScanNet test subset (mAP@25).
 * :fire: February 2023. The source code has been published.
 

This repository contains an implementation of TD3D, a 3D instance segmentation method introduced in our paper:

> **Top-Down Beats Bottom-Up in 3D Instance Segmentation**<br>
> [Maksim Kolodiazhnyi](https://github.com/col14m),
> [Danila Rukhovich](https://github.com/filaPro),
> [Anna Vorontsova](https://github.com/highrut),
> [Anton Konushin](https://scholar.google.com/citations?user=ZT_k-wMAAAAJ)
> <br>
> Samsung AI Center Moscow <br>
> https://arxiv.org/abs/2302.02871

> 
<p align="center"><img src="https://github.com/SamsungLabs/td3d/releases/download/v1.0.0/td3d_scannet.png" alt="drawing" width="90%"/></p>

### Installation
For convenience, we provide a [Dockerfile](docker/Dockerfile).

Alternatively, you can install all required packages manually. This implementation is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) framework.

Please refer to the original installation guide [getting_started.md](docs/en/getting_started.md), including MinkowskiEngine installation, replacing open-mmlab/mmdetection3d with samsunglabs/td3d.

Most of the `TD3D`-related code locates in the following files: 
[detectors/td3d_instance_segmentor.py](mmdet3d/models/detectors/td3d_instance_segmentor.py),
[necks/ngfc_neck.py](mmdet3d/models/necks/ngfc_neck.py),
[decode_heads/td3d_instance_head.py](mmdet3d/models/decode_heads/td3d_instance_head.py).

### Getting Started

Please see [getting_started.md](docs/en/getting_started.md) for basic usage examples.
We follow the `mmdetection3d` data preparation protocol described in [scannet](data/scannet), [s3dis](data/s3dis).


**Training**

To start training, run [train](tools/train.py) with `TD3D` [configs](configs/td3d_is):
```shell
python tools/train.py configs/td3d_is/td3d_is_scannet-3d-18class.py
```

**Testing**

Test pre-trained model using [test](tools/test.py) with `TD3D` [configs](configs/td3d_is). To achieve the best results on Scannet and S3DIS, set `score_thr` to `0.1` and `nms_pre` to `1200` in configs. For Scannet200, set `score_thr` to `0.07` and `nms_pre` to `300`:
```shell
python tools/test.py configs/td3d_is/td3d_is_scannet-3d-18class.py \
    work_dirs/td3d_is_scannet-3d-18class/latest.pth --eval mAP
```

**Visualization**

Visualizations can be created with [test](tools/test.py) script. 
For better visualizations, you may set `score_thr` to `0.20` and `nms_pre` to `200` in configs:
```shell
python tools/test.py configs/td3d_is/td3d_is_scannet-3d-18class.py \
    work_dirs/td3d_is_scannet-3d-18class/latest.pth --eval mAP --show \
    --show-dir work_dirs/td3d_is_scannet-3d-18class
```

### Models (quality on validation subset)

| Dataset | mAP@0.25 | mAP@0.5 | mAP | Download |
|:-------:|:--------:|:-------:|:--------:|:--------:|
| ScanNet | 81.3 | 71.1 | 46.2 | [model](https://github.com/SamsungLabs/td3d/releases/download/v1.0.0/epoch_33.pth) &#124; [config](https://github.com/SamsungLabs/td3d/releases/download/v1.0.0/td3d_is_scannet-3d-18class_public.py) |
| S3DIS (5 area) | 82.8 | 66.5 | 47.4 | [model](https://github.com/SamsungLabs/td3d/releases/download/v1.0.0/epoch_21.pth) &#124; [config](https://github.com/SamsungLabs/td3d/releases/download/v1.0.0/td3d_is_s3dis-3d-5class_public.py) |
| S3DIS (5 area) <br /> (ScanNet pretrain) | 85.6 | 75.5 | 61.1 | [model](https://github.com/SamsungLabs/td3d/releases/download/v1.0.0/epoch_6.pth) &#124; [config](https://github.com/SamsungLabs/td3d/releases/download/v1.0.0/td3d_is_s3dis-3d-5class_public.py) |
| Scannet200 | 39.7 | 33.3 | 22.2 | [model](https://github.com/SamsungLabs/td3d/releases/download/v1.0.0/epoch_28.pth) &#124; [config](https://github.com/SamsungLabs/td3d/releases/download/v1.0.0/td3d_is_scannet200-3d-198class_public.py) |

### Examples

<p align="center"><img src="https://github.com/SamsungLabs/td3d/releases/download/v1.0.0/td3d.png" alt="drawing" width="90%"/></p>

### Citation

If you find this work useful for your research, please cite our paper:
```
@misc{kolodiazhnyi2023topdown,
  doi = {10.48550/ARXIV.2302.02871},
  url = {https://arxiv.org/abs/2302.02871},
  author = {Kolodiazhnyi, Maksim and Rukhovich, Danila and Vorontsova, Anna and Konushin, Anton},
  title = {Top-Down Beats Bottom-Up in 3D Instance Segmentation},
  publisher = {arXiv},
  year = {2023}
}
```
