import logging
from collections import OrderedDict
from typing import List, Tuple

import torch
from mmengine import print_log
from mmengine.model import BaseModule, initialize
from torch import nn
from torch.nn.modules.module import T

VoVNet19_slim_dw_eSE = {
    'stem': [64, 64, 64],
    'stage_conv_ch': [64, 80, 96, 112],
    'stage_out_ch': [112, 256, 384, 512],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "dw": True
}

VoVNet19_dw_eSE = {
    'stem': [64, 64, 64],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "dw": True
}

VoVNet19_slim_eSE = {
    'stem': [64, 64, 128],
    'stage_conv_ch': [64, 80, 96, 112],
    'stage_out_ch': [112, 256, 384, 512],
    'layer_per_block': 3,
    'block_per_stage': [1, 1, 1, 1],
    "dw": False
}

VoVNet19_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "dw": False
}

VoVNet39_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 1, 2, 2],
    "dw": False
}

VoVNet57_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 1, 4, 3],
    "dw": False
}

VoVNet99_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 3, 9, 3],
    "dw": False
}

_STAGE_SPECS = {
    "V-19-slim-dw-eSE": VoVNet19_slim_dw_eSE,
    "V-19-dw-eSE": VoVNet19_dw_eSE,
    "V-19-slim-eSE": VoVNet19_slim_eSE,
    "V-19-eSE": VoVNet19_eSE,
    "V-39-eSE": VoVNet39_eSE,
    "V-57-eSE": VoVNet57_eSE,
    "V-99-eSE": VoVNet99_eSE,
}


def dw_conv3x3(in_channels, out_channels, module_name, postfix,
               stride=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        ('{}_{}/dw_conv3x3'.format(module_name, postfix),
         nn.Conv2d(in_channels, out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   groups=out_channels,
                   bias=False)),
        ('{}_{}/pw_conv1x1'.format(module_name, postfix),
         nn.Conv2d(in_channels, out_channels,
                   kernel_size=1,
                   stride=1,
                   padding=0,
                   groups=1,
                   bias=False)),
        ('{}_{}/pw_norm'.format(module_name, postfix), nn.BatchNorm2d(out_channels)),
        ('{}_{}/pw_relu'.format(module_name, postfix), nn.ReLU(inplace=True)),
    ]


def conv3x3(
        in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=3, padding=1
):
    """3x3 convolution with padding"""
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", nn.BatchNorm2d(out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


def conv1x1(
        in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=1, padding=0
):
    """1x1 convolution with padding"""
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", nn.BatchNorm2d(out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


class eSEModule(nn.Module):
    def __init__(self, channel):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1, padding=0)

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = nn.functional.relu6(x + 3.0, inplace=True) / 6.0
        return input * x


class _OSA_module(nn.Module):
    def __init__(
            self, in_ch, stage_ch, concat_ch, layer_per_block, module_name, identity=False, depth_wise=False
    ):

        super(_OSA_module, self).__init__()

        self.identity = identity
        self.depth_wise = depth_wise
        self.isReduced = False
        self.layers = nn.ModuleList()
        in_channel = in_ch
        if self.depth_wise and in_channel != stage_ch:
            self.isReduced = True
            self.conv_reduction = nn.Sequential(
                OrderedDict(conv1x1(in_channel, stage_ch,
                                    "{}_reduction".format(module_name), "0")))
        for i in range(layer_per_block):
            if self.depth_wise:
                self.layers.append(
                    nn.Sequential(OrderedDict(dw_conv3x3(stage_ch, stage_ch, module_name, i))))
            else:
                self.layers.append(
                    nn.Sequential(OrderedDict(conv3x3(in_channel, stage_ch, module_name, i)))
                )
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(
            OrderedDict(conv1x1(in_channel, concat_ch, module_name, "concat"))
        )

        self.ese = eSEModule(concat_ch)

    def forward(self, x):
        identity_feat = x

        output = [x]
        if self.depthwise and self.isReduced:
            x = self.conv_reduction(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)
        xt = self.ese(xt)
        if self.identity:
            xt = xt + identity_feat
        return xt


class _OSA_stage(nn.Sequential):
    def __init__(
            self, in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num, depth_wise=False
    ):

        super().__init__()
        if not stage_num == 2:
            self.add_module("Pooling", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        module_name = f"OSA{stage_num}_1"
        self.add_module(
            module_name,
            _OSA_module(in_ch, stage_ch, concat_ch, layer_per_block, module_name, depth_wise=depth_wise)
        )
        for i in range(block_per_stage - 1):
            module_name = f"OSA{stage_num}_{i + 2}"
            self.add_module(
                module_name,
                _OSA_module(
                    concat_ch,
                    stage_ch,
                    concat_ch,
                    layer_per_block,
                    module_name,
                    identity=True,
                    depth_wise=depth_wise
                ),
            )


class VoVNet(BaseModule):
    # 仿照原版复现的VoVnet，修改了输出形式，调整了冻结参数的形式
    def __init__(self,
                 spec_name: str,
                 input_ch: int = 3,
                 out_features: List[str] = None,
                 frozen_stages: int = -1,
                 bn_eval: bool = False,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        stage_specs = _STAGE_SPECS[spec_name]

        stem_ch = stage_specs["stem"]
        config_stage_ch = stage_specs["stage_conv_ch"]
        config_concat_ch = stage_specs["stage_out_ch"]
        block_per_stage = stage_specs["block_per_stage"]
        layer_per_block = stage_specs["layer_per_block"]
        depth_wise = stage_specs["dw"]

        self.out_features = out_features

        # Stem module
        conv_type = dw_conv3x3 if depth_wise else conv3x3
        stem = conv3x3(input_ch, stem_ch[0], "stem", "1", 2)
        stem += conv_type(stem_ch[0], stem_ch[1], "stem", "2", 1)
        stem += conv_type(stem_ch[1], stem_ch[2], "stem", "3", 2)
        self.add_module("stem", nn.Sequential((OrderedDict(stem))))
        current_stride = 4
        self._out_feature_strides = {"stem": current_stride, "stage2": current_stride}
        self._out_feature_channels = {"stem": stem_ch[2]}

        # OSA stages
        stem_out_ch = [stem_ch[2]]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]
        self.stage_names = []
        for i in range(4):  # num_stages
            name = "stage%d" % (i + 2)  # stage 2 ... stage 5
            self.stage_names.append(name)
            self.add_module(
                name,
                _OSA_stage(
                    in_ch_list[i],
                    config_stage_ch[i],
                    config_concat_ch[i],
                    block_per_stage[i],
                    layer_per_block,
                    i + 2,
                    depth_wise,
                ),
            )
            self._out_feature_channels[name] = config_concat_ch[i]
            if not i == 0:
                self._out_feature_strides[name] = current_stride = int(current_stride * 2)

        self.init_weights()

    def init_weights(self):
        if not self._is_init:
            if self.init_cfg:
                print_log(
                    f'initialize {self.__class__.__name__} with init_cfg {self.init_cfg}',
                    logger='current',
                    level=logging.DEBUG)
                init_cfg = self.init_cfg
                assert isinstance(init_cfg, dict)
                initialize(self, [init_cfg])
            else:
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight)
            self.is_init = True

    def forward(self, x: torch.Tensor) -> Tuple:
        outputs = []
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs.append(x)
        for name in self.stage_names:
            x = getattr(self, name)(x)
            if name in self._out_features:
                outputs.append(x)

        return tuple(outputs)

    def train(self: T, mode: bool = True) -> T:
        super().train(mode)
        if not mode:
            return self
        # 冻结前层
        if self.frozen_stages >= 0:
            m = getattr(self, 'stem')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
            for i in range(1, self.frozen_stages + 1):
                m = getattr(self, f'stage{i + 1}')
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
        # 冻结BN
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for params in m.parameters():
                        params.requires_grad = False
        return self
