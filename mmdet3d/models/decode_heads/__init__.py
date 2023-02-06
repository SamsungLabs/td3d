# Copyright (c) OpenMMLab. All rights reserved.
from .dgcnn_head import DGCNNHead
from .paconv_head import PAConvHead
from .pointnet2_head import PointNet2Head
from .td3d_instance_head import TD3DInstanceHead

__all__ = ['PointNet2Head', 'DGCNNHead', 'PAConvHead']
