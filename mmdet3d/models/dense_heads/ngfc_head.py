try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

import torch
from torch import nn

from mmcv.runner import BaseModule
from mmcv.cnn import Scale, bias_init_with_prob
from mmcv.ops import nms3d, nms3d_normal
from mmdet.core.bbox.builder import BBOX_ASSIGNERS, build_assigner
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.core.bbox.structures import rotation_3d_in_axis


@HEADS.register_module()
class NgfcOffsetHead(BaseModule):
    def __init__(self,
                 n_classes,
                 in_channels,
                 voxel_size,
                 cls_threshold,
                 assigner,
                 bbox_loss=dict(type='L1Loss'),
                 cls_loss=dict(type='FocalLoss')):
        super(NgfcOffsetHead, self).__init__()
        self.voxel_size = voxel_size
        self.cls_threshold = cls_threshold
        self.assigner = build_assigner(assigner)
        self.bbox_loss = build_loss(bbox_loss)
        self.cls_loss = build_loss(cls_loss)
        self._init_layers(n_classes, in_channels)

    def _init_layers(self, n_classes, in_channels):
        self.bbox_conv = ME.MinkowskiConvolution(
            in_channels, 3, kernel_size=1, bias=True, dimension=3)
        self.cls_conv = ME.MinkowskiConvolution(
            in_channels, n_classes, kernel_size=1, bias=True, dimension=3)
        self.conv = nn.Conv1d(in_channels, in_channels, 1)

    def init_weights(self):
        nn.init.normal_(self.bbox_conv.kernel, std=.01)
        nn.init.normal_(self.cls_conv.kernel, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))

    def forward(self, x):
        # -> bbox_preds, cls_preds, points, sampled and shifted tensor
        bbox_pred = self.bbox_conv(x)
        cls_pred = self.cls_conv(x)

        bbox_preds, cls_preds, points = [], [], []
        for permutation in x.decomposition_permutations:
            bbox_preds.append(bbox_pred.features[permutation])
            cls_preds.append(cls_pred.features[permutation])
            points.append(x.coordinates[permutation][:, 1:] * self.voxel_size)

        mask = cls_pred.features.max(dim=1).values.sigmoid() > self.cls_threshold
        coordinates = x.coordinates[mask]
        features = x.features[mask]
        shifts = bbox_pred.features[mask]
        new_coordinates = torch.cat((
            coordinates[:, :1],
            ((coordinates[:, 1:] * self.voxel_size
              + shifts) / self.voxel_size).round()), dim=1)  # todo: .int() ?


        if features.shape[0] > 0:
            features = self.conv(features.unsqueeze(2))[:, :, 0]
        new_coordinates = torch.cat((x.coordinates, new_coordinates))
        features = torch.cat((x.features, features))

        # SparseTensor with initial voxel size and stride = 1
        x = ME.SparseTensor(
            coordinates=new_coordinates,
            features=features,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
        return bbox_preds, cls_preds, points, x

    # per scene
    def _loss_single(self,
                     bbox_preds,
                     cls_preds,
                     points,
                     gt_bboxes,
                     gt_labels,
                     img_meta):
        assigned_ids = self.assigner.assign([points], gt_bboxes)

        # cls loss
        n_classes = cls_preds.shape[1]
        pos_mask = assigned_ids >= 0
        cls_targets = torch.where(pos_mask, gt_labels[assigned_ids], n_classes)
        avg_factor = max(pos_mask.sum(), 1)
        cls_loss = self.cls_loss(cls_preds, cls_targets, avg_factor=avg_factor)

        # bbox loss
        pos_bbox_preds = bbox_preds[pos_mask]
        if pos_mask.sum() > 0:
            pos_points = points[pos_mask]
            pos_bbox_preds = bbox_preds[pos_mask]
            bbox_targets = gt_bboxes.gravity_center.to(points.device)
            pos_bbox_targets = bbox_targets[assigned_ids][pos_mask] - pos_points
            bbox_loss = self.bbox_loss(pos_bbox_preds, pos_bbox_targets,
                                       avg_factor=pos_bbox_targets.abs().sum())
        else:
            bbox_loss = pos_bbox_preds.sum()
        return bbox_loss, cls_loss

    def _loss(self, bbox_preds, cls_preds, points,
              gt_bboxes, gt_labels, img_metas):
        bbox_losses, cls_losses = [], []
        for i in range(len(img_metas)):
            bbox_loss, cls_loss = self._loss_single(
                bbox_preds=bbox_preds[i],
                cls_preds=cls_preds[i],
                points=points[i],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i])
            bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
        return dict(
            offset_loss=torch.mean(torch.stack(bbox_losses)),
            obj_loss=torch.mean(torch.stack(cls_losses)))

    def forward_train(self, x, gt_bboxes, gt_labels, img_metas):
        bbox_preds, cls_preds, points, x = self(x)
        return x, self._loss(bbox_preds, cls_preds, points,
                             gt_bboxes, gt_labels, img_metas)

    def forward_test(self, x, img_metas):
        _, _, _, x = self(x)
        return x


@HEADS.register_module()
class NgfcHead(BaseModule):
    def __init__(self,
                 n_classes,
                 in_channels,
                 n_levels,
                 n_reg_outs,
                 padding,
                 voxel_size,
                 assigner,
                 bbox_loss=dict(type='AxisAlignedIoULoss'),
                 cls_loss=dict(type='FocalLoss'),
                 train_cfg=None,
                 test_cfg=None):
        super(NgfcHead, self).__init__()
        self.padding = padding
        self.voxel_size = voxel_size
        self.assigner = build_assigner(assigner)
        self.bbox_loss = build_loss(bbox_loss)
        self.cls_loss = build_loss(cls_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(n_classes, in_channels, n_levels, n_reg_outs)

    def _init_layers(self, n_classes, in_channels, n_levels, n_reg_outs):
        for i in range(n_levels):
            self.__setattr__(f'scale_{i}', Scale(1.))
        self.bbox_conv = ME.MinkowskiConvolution(
            in_channels, n_reg_outs, kernel_size=1, bias=True, dimension=3)
        self.cls_conv = ME.MinkowskiConvolution(
            in_channels, n_classes, kernel_size=1, bias=True, dimension=3)

    def init_weights(self):
        nn.init.normal_(self.bbox_conv.kernel, std=.01)
        nn.init.normal_(self.cls_conv.kernel, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))

    # per level
    def _forward_single(self, x, scale):
        reg_final = self.bbox_conv(x).features
        reg_distance = torch.exp(scale(reg_final[:, :6]))
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_distance, reg_angle), dim=1)
        cls_pred = self.cls_conv(x).features

        bbox_preds, cls_preds, points = [], [], []
        for permutation in x.decomposition_permutations:
            bbox_preds.append(bbox_pred[permutation])
            cls_preds.append(cls_pred[permutation])
            points.append(x.coordinates[permutation][:, 1:] * self.voxel_size)

        return bbox_preds, cls_preds, points

    def forward(self, x):
        bbox_preds, cls_preds, points = [], [], []
        for i in range(len(x)):
            bbox_pred, cls_pred, point = self._forward_single(
                x[i], self.__getattr__(f'scale_{i}'))
            bbox_preds.append(bbox_pred)
            cls_preds.append(cls_pred)
            points.append(point)
        return bbox_preds, cls_preds, points

    @staticmethod
    def _bbox_to_loss(bbox):
        """Transform box to the axis-aligned or rotated iou loss format.
        Args:
            bbox (Tensor): 3D box of shape (N, 6) or (N, 7).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        # rotated iou loss accepts (x, y, z, w, h, l, heading)
        if bbox.shape[-1] != 6:
            return bbox

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)

    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.
        Args:
            points (Tensor): Final locations of shape (N, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
                or (N, 8).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + (bbox_pred[:, 1] - bbox_pred[:, 0]) / 2
        y_center = points[:, 1] + (bbox_pred[:, 3] - bbox_pred[:, 2]) / 2
        z_center = points[:, 2] + (bbox_pred[:, 5] - bbox_pred[:, 4]) / 2

        # dx_min, dx_max, dy_min, dy_max, dz_min, dz_max -> x, y, z, w, l, h
        base_bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 0] + bbox_pred[:, 1],
            bbox_pred[:, 2] + bbox_pred[:, 3],
            bbox_pred[:, 4] + bbox_pred[:, 5],
        ], -1)

        # axis-aligned case
        if bbox_pred.shape[1] == 6:
            return base_bbox

        # rotated case: ..., sin(2a)ln(q), cos(2a)ln(q)
        scale = bbox_pred[:, 0] + bbox_pred[:, 1] + \
                bbox_pred[:, 2] + bbox_pred[:, 3]
        q = torch.exp(
            torch.sqrt(
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
        return torch.stack(
            (x_center, y_center, z_center, scale / (1 + q), scale /
             (1 + q) * q, bbox_pred[:, 5] + bbox_pred[:, 4], alpha),
            dim=-1)

    # per scene
    def _loss_single(self,
                     bbox_preds,
                     cls_preds,
                     points,
                     gt_bboxes,
                     gt_labels,
                     img_meta):
        assigned_ids = self.assigner.assign(points, gt_bboxes)
        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        points = torch.cat(points)

        # cls loss
        n_classes = cls_preds.shape[1]
        pos_mask = assigned_ids >= 0
        cls_targets = torch.where(pos_mask, gt_labels[assigned_ids], n_classes)
        avg_factor = max(pos_mask.sum(), 1)
        cls_loss = self.cls_loss(cls_preds, cls_targets, avg_factor=avg_factor)

        # bbox loss
        pos_bbox_preds = bbox_preds[pos_mask]
        if pos_mask.sum() > 0:
            pos_points = points[pos_mask]
            pos_bbox_preds = bbox_preds[pos_mask]
            bbox_targets = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            pos_bbox_targets = bbox_targets.to(points.device)[assigned_ids][pos_mask]
            pos_bbox_targets = torch.cat((
                pos_bbox_targets[:, :3],
                pos_bbox_targets[:, 3:6] + self.padding,
                pos_bbox_targets[:, 6:]), dim=1)
            if pos_bbox_preds.shape[1] == 6:
                pos_bbox_targets = pos_bbox_targets[:, :6]
            bbox_loss = self.bbox_loss(
                self._bbox_to_loss(
                    self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)),
                self._bbox_to_loss(pos_bbox_targets))
        else:
            bbox_loss = pos_bbox_preds.sum()
        return bbox_loss, cls_loss

    def _loss(self, bbox_preds, cls_preds, points,
              gt_bboxes, gt_labels, img_metas):
        bbox_losses, cls_losses = [], []
        for i in range(len(img_metas)):
            bbox_loss, cls_loss = self._loss_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i])
            bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
        return dict(
            bbox_loss=torch.mean(torch.stack(bbox_losses)),
            cls_loss=torch.mean(torch.stack(cls_losses)))

    def forward_train(self, x, gt_bboxes, gt_labels, img_metas):
        bbox_preds, cls_preds, points = self(x)
        return self._loss(bbox_preds, cls_preds, points,
                          gt_bboxes, gt_labels, img_metas)

    def _nms(self, bboxes, scores, img_meta):
        """Multi-class nms for a single scene.
        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            img_meta (dict): Scene meta data.
        Returns:
            Tensor: Predicted bboxes.
            Tensor: Predicted scores.
            Tensor: Predicted labels.
        """
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                    dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores,
                                   self.test_cfg.iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))

        if yaw_flag:
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels

    def _get_bboxes_single(self, bbox_preds, cls_preds, points, img_meta):
        scores = torch.cat(cls_preds).sigmoid()
        bbox_preds = torch.cat(bbox_preds)
        points = torch.cat(points)
        max_scores, _ = scores.max(dim=1)

        if len(scores) > self.test_cfg.nms_pre > 0:
            _, ids = max_scores.topk(self.test_cfg.nms_pre)
            bbox_preds = bbox_preds[ids]
            scores = scores[ids]
            points = points[ids]

        boxes = self._bbox_pred_to_bbox(points, bbox_preds)
        boxes = torch.cat((
            boxes[:, :3],
            boxes[:, 3:6] - self.padding,
            boxes[:, 6:]), dim=1)
        boxes, scores, labels = self._nms(boxes, scores, img_meta)
        return boxes, scores, labels

    def _get_bboxes(self, bbox_preds, cls_preds, points, img_metas):
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i])
            results.append(result)
        return results

    def forward_test(self, x, img_metas):
        bbox_preds, cls_preds, points = self(x)
        return self._get_bboxes(bbox_preds, cls_preds, points, img_metas)


@BBOX_ASSIGNERS.register_module()
class NgfcAssigner:
    def __init__(self, min_pts_threshold, top_pts_threshold, padding):
        # min_pts_threshold: per level
        # top_pts_threshold: per box
        self.min_pts_threshold = min_pts_threshold
        self.top_pts_threshold = top_pts_threshold
        self.padding = padding

    @torch.no_grad()
    def assign(self, points, gt_bboxes):
        # -> object id or -1 for each point
        float_max = points[0].new_tensor(1e8)
        n_levels = len(points)
        levels = torch.cat([points[i].new_tensor(i, dtype=torch.long).expand(len(points[i]))
                            for i in range(len(points))])
        points = torch.cat(points)
        n_points = len(points)
        n_boxes = len(gt_bboxes)
        volumes = gt_bboxes.volume.to(points.device).unsqueeze(0).expand(n_points, n_boxes)

        # condition 1: point inside enlarged box
        boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
        boxes = boxes.to(points.device).expand(n_points, n_boxes, 7)
        boxes = torch.cat((boxes[..., :3], boxes[..., 3:6] + self.padding, boxes[..., 6:]), dim=-1)
        points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
        face_distances = get_face_distances(points, boxes)
        inside_box_condition = face_distances.min(dim=-1).values > 0
        # print(gt_bboxes.tensor)
        # for i in range(n_levels):
        #     print(i, inside_box_condition[levels == i].sum(dim=0))

        # condition 2: positive points per level >= limit
        # calculate positive points per level
        n_pos_points_per_level = []
        for i in range(n_levels):
            n_pos_points_per_level.append(torch.sum(inside_box_condition[levels == i], dim=0))
        # find best level
        n_pos_points_per_scale = torch.stack(n_pos_points_per_level, dim=0)
        lower_limit_mask = n_pos_points_per_scale < self.min_pts_threshold
        lower_index = torch.argmax(lower_limit_mask.int(), dim=0) - 1
        lower_index = torch.where(lower_index < 0, 0, lower_index)
        all_upper_limit_mask = torch.all(torch.logical_not(lower_limit_mask), dim=0)
        best_level = torch.where(all_upper_limit_mask, n_levels - 1, lower_index)
        # keep only points with best level
        best_level = torch.unsqueeze(best_level, 0).expand(n_points, n_boxes)
        levels = torch.unsqueeze(levels, 1).expand(n_points, n_boxes)
        level_condition = best_level == levels

        # condition 3: keep topk location per box by center distance
        center_distances = torch.sum(torch.pow(boxes[..., :3] - points, 2), dim=-1)
        center_distances = torch.where(inside_box_condition, center_distances, float_max)
        center_distances = torch.where(level_condition, center_distances, float_max)
        topk_distances = torch.topk(center_distances,
                                    min(self.top_pts_threshold + 1, len(center_distances)),
                                    largest=False, dim=0).values[-1]
        topk_condition = center_distances < topk_distances.unsqueeze(0)

        # condition 4: min volume box per point
        volumes = torch.where(inside_box_condition, volumes, torch.ones_like(volumes) * float_max)
        volumes = torch.where(level_condition, volumes, torch.ones_like(volumes) * float_max)
        volumes = torch.where(topk_condition, volumes, torch.ones_like(volumes) * float_max)
        min_volumes, min_ids = volumes.min(dim=1)
        min_inds = torch.where(min_volumes < float_max, min_ids, -1)
        return min_inds


def get_face_distances(points, boxes):
    # points: of shape (..., 3)
    # boxes: of shape (..., 7)
    # -> of shape (..., 6): dx_min, dx_max, dy_min, dy_max, dz_min, dz_max
    shift = torch.stack((
        points[..., 0] - boxes[..., 0],
        points[..., 1] - boxes[..., 1],
        points[..., 2] - boxes[..., 2]), dim=-1).permute(1, 0, 2)
    shift = rotation_3d_in_axis(shift, -boxes[0, :, 6], axis=2).permute(1, 0, 2)
    centers = boxes[..., :3] + shift
    dx_min = centers[..., 0] - boxes[..., 0] + boxes[..., 3] / 2
    dx_max = boxes[..., 0] + boxes[..., 3] / 2 - centers[..., 0]
    dy_min = centers[..., 1] - boxes[..., 1] + boxes[..., 4] / 2
    dy_max = boxes[..., 1] + boxes[..., 4] / 2 - centers[..., 1]
    dz_min = centers[..., 2] - boxes[..., 2] + boxes[..., 5] / 2
    dz_max = boxes[..., 2] + boxes[..., 5] / 2 - centers[..., 2]
    return torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max), dim=-1)
