from .builder import DATASETS
from mmdet3d.datasets.scannet_dataset import ScanNetInstanceSegV2Dataset

@DATASETS.register_module()
class Structured3DDataset(ScanNetInstanceSegV2Dataset):
    r"""S3DIS Dataset for Detection Task.

    This class is the inner dataset for S3DIS. Since S3DIS has 6 areas, we
    often train on 5 of them and test on the remaining one. The one for
    test is Area_5 as suggested in `GSDN <https://arxiv.org/abs/2006.12356>`_.
    To concatenate 5 areas during training
    `mmdet.datasets.dataset_wrappers.ConcatDataset` should be used.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Depth' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """
    CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'picture', 'counter', 
                    'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'fridge', 'television', 
                    'night stand', 'toilet', 'sink', 'lamp', 'bathtub', 'structure', 'furniture', 'prop')

    VALID_CLASS_IDS = (3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19, 24, 25, 32, 33, 34, 35, 36, 38, 39, 40)