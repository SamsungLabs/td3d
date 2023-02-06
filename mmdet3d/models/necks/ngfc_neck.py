try:
    import MinkowskiEngine as ME
    from MinkowskiEngine.modules.resnet_block import BasicBlock
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

from torch import nn

from mmcv.runner import BaseModule
from mmdet3d.models.builder import NECKS


@NECKS.register_module()
class NgfcNeck(BaseModule):
    def __init__(self, in_channels):
        super(NgfcNeck, self).__init__()
        self._init_layers(in_channels)

    def _init_layers(self, in_channels):
        for i in range(len(in_channels)):
            if i > 0:
                self.__setattr__(f'up_block_{i}', make_up_block(in_channels[i], in_channels[i - 1]))
            if i < len(in_channels) - 1:
                self.__setattr__(f'lateral_block_{i}',
                                 make_block(in_channels[i], in_channels[i]))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x):
        inputs = x
        x = inputs[-1]
        for i in range(len(inputs) - 1, -1, -1):
            if i < len(inputs) - 1:
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                # print('NgfcNeck', i, x.features.shape, inputs[i].features.shape)
                x = inputs[i] + x
                x = self.__getattr__(f'lateral_block_{i}')(x)
        return x


@NECKS.register_module()
class NgfcTinyNeck(BaseModule):
    def __init__(self, in_channels, out_channels):
        super(NgfcTinyNeck, self).__init__()
        self._init_layers(in_channels, out_channels)

    def _init_layers(self, in_channels, out_channels):
        for i in range(len(in_channels)):
            if i > 0:
                self.__setattr__(
                    f'up_block_{i}',
                    make_up_block(in_channels[i], in_channels[i - 1]))
            if i < len(in_channels) - 1:
                self.__setattr__(
                    f'lateral_block_{i}',
                    make_block(in_channels[i], in_channels[i]))
            self.__setattr__(
                f'out_block_{i}',
                make_block(in_channels[i], out_channels))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x):
        outs = []
        inputs = x
        x = inputs[-1]
        for i in range(len(inputs) - 1, -1, -1):
            if i < len(inputs) - 1:
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                # print('NgfcTinyNeck', i, x.features.shape, inputs[i].features.shape)
                x = inputs[i] + x
                x = self.__getattr__(f'lateral_block_{i}')(x)
            out = self.__getattr__(f'out_block_{i}')(x)
            outs.append(out)
        return outs[::-1]


@NECKS.register_module()
class NgfcTinySegmentationNeck(BaseModule):
    def __init__(self, in_channels, out_channels):
        super(NgfcTinySegmentationNeck, self).__init__()
        self._init_layers(in_channels, out_channels)
        
        self.upsample_st_4 = nn.Sequential(
                        ME.MinkowskiConvolutionTranspose(
                            128,
                            64,
                            kernel_size=3,
                            stride=4,
                            dimension=3),
                        ME.MinkowskiBatchNorm(64),
                        ME.MinkowskiReLU(inplace=True))
        
        self.conv_32_ch = nn.Sequential(
                        ME.MinkowskiConvolution(
                            64,
                            32,
                            kernel_size=3,
                            stride=1,
                            dimension=3),
                        ME.MinkowskiBatchNorm(32),
                        ME.MinkowskiReLU(inplace=True))

    def _init_layers(self, in_channels, out_channels):
        for i in range(len(in_channels)):
            if i > 0:
                self.__setattr__(
                    f'up_block_{i}',
                    make_up_block(in_channels[i], in_channels[i - 1]))
            if i < len(in_channels) - 1:
                self.__setattr__(
                    f'lateral_block_{i}',
                    make_block(in_channels[i], in_channels[i]))
            self.__setattr__(
                f'out_block_{i}',
                make_block(in_channels[i], out_channels))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x):
        feats_st_2 = x[0]
        outs = []
        inputs = x[1:]
        x = inputs[-1]
        for i in range(len(inputs) - 1, -1, -1):
            if i < len(inputs) - 1:
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                x = inputs[i] + x
                x = self.__getattr__(f'lateral_block_{i}')(x)
            out = self.__getattr__(f'out_block_{i}')(x)
            outs.append(out)
        
        outs = outs[::-1]

        seg_feats = self.conv_32_ch(self.upsample_st_4(outs[0]) + feats_st_2)
        return [seg_feats] + outs


class BiFPNLayer(BaseModule):
    def __init__(self, n_channels, n_levels):
        super(BiFPNLayer, self).__init__()
        self._init_layers(n_channels, n_levels)

    def _init_layers(self, n_channels, n_levels):
        for i in range(n_levels):
            if i > 0:
                self.__setattr__(
                    f'up_block_{i}',
                    make_up_block(n_channels, n_channels))
                self.__setattr__(
                    f'down_block_{i}',
                    make_up_block(n_channels, n_channels))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x):
        x1s = []
        inputs = x
        x = inputs[-1]
        for i in range(len(inputs) - 1, -1, -1):
            if i < len(inputs) - 1:
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                x = inputs[i] + x
            x1s.append(x)
        x1s = x1s[::-1]
        x2s = [x]
        for i in range(1, len(inputs)):
            x = self.__getattr__(f'down_block_{i}')(x)
            x = x + inputs[i]
            if i < len(inputs) - 1:
                x = x + x1s[i]
            x2s.append(x)
        return x2s


@NECKS.register_module()
class BiFPNNeck(BaseModule):
    def __init__(self, in_channels, out_channels, n_blocks):
        super(BiFPNNeck, self).__init__()
        self.n_levels = len(in_channels)
        self.n_blocks = n_blocks
        self._init_layers(in_channels, out_channels, n_blocks)

    def _init_layers(self, in_channels, out_channels):
        for i in range(len(in_channels)):
            self.__setattr__(
                f'in_block_{i}',
                make_block(in_channels[i], out_channels, 1))
        for i in range(self.n_blocks):
            self.__setattr__(
                f'block_{i}',
                BiFPNLayer(out_channels, self.n_levels))


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x):
        pass  # todo: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def make_block(in_channels, out_channels, kernel_size=3):
    return nn.Sequential(
        ME.MinkowskiConvolution(in_channels, out_channels,
                                kernel_size=kernel_size, dimension=3),
        ME.MinkowskiBatchNorm(out_channels),
        ME.MinkowskiReLU(inplace=True))


def make_down_block(in_channels, out_channels):
    return nn.Sequential(
        ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3,
                                stride=2, dimension=3),
        ME.MinkowskiBatchNorm(out_channels),
        ME.MinkowskiReLU(inplace=True))


def make_up_block(in_channels, out_channels):
    return nn.Sequential(
        ME.MinkowskiConvolutionTranspose(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            dimension=3),
        ME.MinkowskiBatchNorm(out_channels),
        ME.MinkowskiReLU(inplace=True))
