try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

from mmdet3d.models import DETECTORS, build_backbone, build_neck, build_head
from .base import Base3DDetector
import torch

@DETECTORS.register_module()
class TD3DInstanceSegmentor(Base3DDetector):
    r"""Two-stage instance segmentor based on MinkowskiEngine.
    The first stage is bbox detector. The second stage is two-class pointwise segmentor (foreground/background).

    Args:
        backbone (dict): Config of the backbone.
        neck (dict): Config of the neck.
        head (dict): Config of the head.
        voxel_size (float): Voxel size in meters.
        train_cfg (dict, optional): Config for train stage. Defaults to None.
        test_cfg (dict, optional): Config for test stage. Defaults to None.
        init_cfg (dict, optional): Config for weight initialization.
            Defaults to None.
        pretrained (str, optional): Deprecated initialization parameter.
            Defaults to None.
    """
    def __init__(self,
                 backbone,
                 neck,
                 head,
                 voxel_size,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(TD3DInstanceSegmentor, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        head.update(train_cfg=train_cfg)
        head.update(test_cfg=test_cfg)
        self.head = build_head(head)
        self.voxel_size = voxel_size
        self.init_weights()

    def extract_feat(self, points):
        """Extract features from points.

        Args:
            points (list[Tensor]): Raw point clouds.

        Returns:
            SparseTensor: Voxelized point clouds.
        """
        x = self.backbone(points)
        x = self.neck(x)
        return x   
    
    def collate(self, points, quantization_mode):
        coordinates, features = ME.utils.batch_sparse_collate(
            [(p[:, :3] / self.voxel_size, p[:, 3:]) for p in points],
            dtype=points[0].dtype,
            device=points[0].device)
        return ME.TensorField(
            features=features,
            coordinates=coordinates,
            quantization_mode=quantization_mode,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=points[0].device,
        )

    def forward_train(self, points, gt_bboxes_3d, gt_labels_3d,
                      pts_semantic_mask, pts_instance_mask, img_metas):
        """Forward of training.

        Args:
            points (list[Tensor]): Raw point clouds.
            gt_bboxes_3d (list[BaseInstance3DBoxes]): Ground truth
                bboxes of each sample.
            gt_labels_3d (list[torch.Tensor]): Labels of each sample.
            pts_semantic_mask (list[torch.Tensor]): Per point semantic labels
                of each sample.
            pts_instance_mask (list[torch.Tensor]): Per point instance labels
                of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            dict: Loss values.
        """
        
        # points = [torch.cat([p, torch.unsqueeze(m, 1)], dim=1) for p, m in zip(points, pts_instance_mask)]
        points = [torch.cat([p, torch.unsqueeze(inst, 1), torch.unsqueeze(sem, 1)], dim=1) for p, inst, sem in zip(points, pts_instance_mask, pts_semantic_mask)]
        field = self.collate(points, ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
        x = field.sparse()
        targets = x.features[:, 3:].round().long()
        x = ME.SparseTensor(
            x.features[:, :3],
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        x = self.extract_feat(x)
        
        losses = self.head.forward_train(x, targets, field, gt_bboxes_3d, gt_labels_3d,
                                         pts_semantic_mask, pts_instance_mask, img_metas)
        return losses

    def simple_test(self, points, img_metas, *args, **kwargs):
        """Test without augmentations.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            list[dict]: Predicted 3d instances.
        """

        field = self.collate(points, ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
        x = self.extract_feat(field.sparse())
        
        instances = self.head.forward_test(x, field, img_metas)
        results = []
        for mask, label, score in instances:
            results.append(dict(
                instance_mask=mask.cpu(),
                instance_label=label.cpu(),
                instance_score=score.cpu()))
        return results

    def aug_test(self, points, img_metas, **kwargs):
        """Test with augmentations.

        Args:
            points (list[list[torch.Tensor]]): Points of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        raise NotImplementedError
