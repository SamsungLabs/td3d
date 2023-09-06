### Prepare STPLS3D Data

We follow the procedure in [Mask3D](https://github.com/JonasSchult/Mask3D).

1. Download STPLS3D data by filling this [Google form](https://docs.google.com/forms/d/e/1FAIpQLSf0jsHw4Q6FFB6AjEgTkF2tgHdMMFyLjC-7fDHrmV01Kci0aA/viewform). Download the `Synthetic_v3_InstanceSegmentation.zip` file and unzip it. Link or move the folder to this level of directory.

2. In this directory, extract point clouds and annotations by running `python prepare_data_inst_instance_stpls3d.py`.

The directory structure after pre-processing should be as below

```
stpls3d
├── prepare_data_inst_instance_stpls3d.py
├── README.md
├── Synthetic_v3_InstanceSegmentation
├── points
│   ├── xxxxx.bin
├── instance_mask
│   ├── xxxxx.bin
├── semantic_mask
│   ├── xxxxx.bin
├── stpls3d_infos_train.pkl
├── stpls3d_infos_val.pkl
```
