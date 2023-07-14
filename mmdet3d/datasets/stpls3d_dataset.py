# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp

import numpy as np

from .builder import DATASETS
from .scannet_dataset import ScanNetInstanceSegV2Dataset

@DATASETS.register_module()
class STPLS3DInstanceSegDataset(ScanNetInstanceSegV2Dataset):
    CLASSES = ("building", "low vegetation", "med. vegetation", "high vegetation", "vehicle", "truck", "aircraft", 
                "militaryVehicle", "bike", "motorcycle", "light pole", "street sign", "clutter", "fence")
    VALID_CLASS_IDS = tuple(range(1, len(CLASSES) + 1))

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - img_prefix (str, optional): Prefix of image files.
                - img_info (dict, optional): Image info.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        pts_filename = osp.join(self.data_root, info['pts_path'])
        input_dict = dict(sample_idx=sample_idx)

        if self.modality['use_depth']:
            input_dict['pts_filename'] = pts_filename
            input_dict['file_name'] = pts_filename

        if self.modality['use_camera']:
            img_info = []
            for img_path in info['img_paths']:
                img_info.append(
                    dict(filename=osp.join(self.data_root, img_path)))
            intrinsic = info['intrinsics']
            axis_align_matrix = self._get_axis_align_matrix(info)
            depth2img = []
            for extrinsic in info['extrinsics']:
                depth2img.append(
                    intrinsic @ np.linalg.inv(axis_align_matrix @ extrinsic))

            input_dict['img_prefix'] = None
            input_dict['img_info'] = img_info
            input_dict['depth2img'] = depth2img

        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos
        return input_dict
