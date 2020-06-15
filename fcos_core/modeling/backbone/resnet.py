# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Variant of the resnet module that takes cfg as an argument.
Example usage. Strings may be specified in the config file.
    model = ResNet(
        "StemWithFixedBatchNorm",
        "BottleneckWithFixedBatchNorm",
        "ResNet50StagesTo4",
    )
OR:
    model = ResNet(
        "StemWithGN",
        "BottleneckWithGN",
        "ResNet50StagesTo4",
    )
Custom implementations may be written in user code and hooked in via the
`register_*` functions.
"""
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from fcos_core.layers import FrozenBatchNorm2d
from fcos_core.layers import Conv2d
from fcos_core.modeling.make_layers import group_norm
from fcos_core.utils.registry import Registry


# ResNet stage specification 通过一个命名元组来设定resnet各阶段的参数
StageSpec = namedtuple(
    "StageSpec",
    [
        "index",  # Index of the stage, eg 1, 2, ..,. 5
        "block_count",  # Number of residual blocks in the stage
        "return_features",  # True => return the last feature map from this stage
    ],
)

# -----------------------------------------------------------------------------
# Standard ResNet models
# -----------------------------------------------------------------------------
#一下的元组会通过_STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]来选定，我只放了resnet50的
# ResNet-50 (including all stages)
ResNet50StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, False), (4, 3, True))
)
# ResNet-50 up to stage 4 (excludes stage 5) 只使用到第四阶段输出的特征图
ResNet50StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, True))
)
# ResNet-101 (including all stages)
ResNet101StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, False), (4, 3, True))
)
# ResNet-101 up to stage 4 (excludes stage 5)
ResNet101StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, True))
)
# ResNet-50-FPN (including all stages)
#由于fpn需要用到每一个阶段输出的特征图, 故return_features参数均为True
#这里的index 1 2 3 4　对应这C2 C3 C4 C5
ResNet50FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 6, True), (4, 3, True))
)
# ResNet-101-FPN (including all stages)
ResNet101FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 23, True), (4, 3, True))
)
# ResNet-152-FPN (including all stages)
ResNet152FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 8, True), (3, 36, True), (4, 3, True))
)

class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()

        # If we want to use the cfg in forward(), then we should make a copy
        # of it and store it for later use:
        # self.cfg = cfg.clone()

        # Translate string names to implementations
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC] #cfg.MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY] #cfg.MODEL.BACKBONE.CONV_BODY = "R-50-C4"
        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC] #cfg.MODEL.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"

        # Construct the stem module 此处是stem的实现，即resnet的第一阶段的conv1
        self.stem = stem_module(cfg)

        # Constuct the specified ResNet stages
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS # 1
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP #64
        in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS #64
        stage2_bottleneck_channels = num_groups * width_per_group # 1*64
        stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS # 256
        self.stages = []
        self.return_features = {} #定义字典
        for stage_spec in stage_specs: #
            name = "layer" + str(stage_spec.index)
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            #每经过一个stage, bottleneck_channels和out_channels就会翻倍
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            out_channels = stage2_out_channels * stage2_relative_factor
            # 循环调用_make_stage，依次实现conv2_x~conv5_x
            module = _make_stage(
                transformation_module,  # BottleneckWithFixedBatchNorm
                in_channels,
                bottleneck_channels,
                out_channels,
                stage_spec.block_count,
                num_groups,
                cfg.MODEL.RESNETS.STRIDE_IN_1X1,
                first_stride=int(stage_spec.index > 1) + 1, # 当处于stage3~5时, 使用stride=2来downsize
            )
            in_channels = out_channels
            self.add_module(name, module)
            self.stages.append(name)
            self.return_features[name] = stage_spec.return_features

        # Optionally freeze (requires_grad=False) parts of the backbone
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        # 根据给定的freeze_at参数冻结相应层的参数更新
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "layer" + str(stage_index))
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            # 将stage2~5中需要返回的某些层的特征图以列表形式保存，作为FPN的输入
            if self.return_features[stage_name]:
                outputs.append(x)
        return outputs


class ResNetHead(nn.Module):
    def __init__(
        self,
        block_module,
        stages,
        num_groups=1,
        width_per_group=64,
        stride_in_1x1=True,
        stride_init=None,
        res2_out_channels=256,
        dilation=1
    ):
        super(ResNetHead, self).__init__()

        stage2_relative_factor = 2 ** (stages[0].index - 1)
        stage2_bottleneck_channels = num_groups * width_per_group
        out_channels = res2_out_channels * stage2_relative_factor
        in_channels = out_channels // 2
        bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor

        block_module = _TRANSFORMATION_MODULES[block_module]

        self.stages = []
        stride = stride_init
        for stage in stages:
            name = "layer" + str(stage.index)
            if not stride:
                stride = int(stage.index > 1) + 1
            module = _make_stage(
                block_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage.block_count,
                num_groups,
                stride_in_1x1,
                first_stride=stride,
                dilation=dilation
            )
            stride = None
            self.add_module(name, module)
            self.stages.append(name)
        self.out_channels = out_channels

    def forward(self, x):
        for stage in self.stages:
            x = getattr(self, stage)(x)
        return x


def _make_stage(
    transformation_module,
    in_channels,
    bottleneck_channels,
    out_channels,
    block_count,
    num_groups,
    stride_in_1x1,
    first_stride,
    dilation=1
):
    blocks = []
    stride = first_stride
    # 循环调用类Bottleneck，每调用一次构造一个瓶颈结构
    for _ in range(block_count):
        blocks.append(
            transformation_module(
                in_channels,
                bottleneck_channels,
                out_channels,
                num_groups,
                stride_in_1x1,
                stride,
                dilation=dilation
            )
        )
        stride = 1 # 注意就是第一次的stride=first_stride，之后都等于1
        in_channels = out_channels
    return nn.Sequential(*blocks)

#resnet的瓶颈结构
class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels, # bottleneck的输入channels
        bottleneck_channels, # bottleneck压缩后的channels
        out_channels, # bottleneck的输出channels
        num_groups, # bottleneck分组的num
        stride_in_1x1,  # 在每个stage的开始的1x1conv中的stride
        stride, # 卷积步长
        dilation,  # 空洞卷积的间隔
        norm_func # 用哪一个归一化函数
    ):
        super(Bottleneck, self).__init__()

        # downsample: 当 bottleneck 的输入和输出的 channels 不相等时, 则需要采用一定的策略
        # 在原文中, 有 A, B, C三种策略, 本文采用的是 B 策略(也是原文推荐的)
        # 即只有在输入输出通道数不相等时才使用 projection shortcuts,
        # 也就是利用参数矩阵映射使得输入输出的 channels 相等
        self.downsample = None
        # 当输入输出通道数不同时, 额外添加一个1×1的卷积层使得输入通道数映射成输出通道数
        if in_channels != out_channels:
            down_stride = stride if dilation == 1 else 1
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=down_stride, bias=False
                ),
                norm_func(out_channels),
            )
            for modules in [self.downsample,]:
                for l in modules.modules():
                    if isinstance(l, Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)

        if dilation > 1:
            stride = 1 # reset to be 1

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        # 这里的意思就是本来论文里的stride=2的卷积用在stage3-5的第一个1x1conv上，现在用在
        # 3x3conv里，但是这里因为是原来框架的，我打印出来还是在1x1conv上，系没有删除注释
        # 因为下面调用的时候都是stride_in_1x1=True
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )
        self.bn1 = norm_func(bottleneck_channels)
        # TODO: specify init for the above
        #dcn_config字典中有键"stage_with_dcn"，则返回对应的值，否则为False

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation
        )
        self.bn2 = norm_func(bottleneck_channels)
        # 创建bottleneck的第3层卷积层
        self.conv3 = Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = norm_func(out_channels)

        for l in [self.conv1, self.conv2, self.conv3,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x): #前向传播
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)

        out0 = self.conv3(out)
        out = self.bn3(out0)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  #跳连结构，add起来
        out = F.relu_(out)

        return out

#resnet的基础茎干网络
class BaseStem(nn.Module):
    def __init__(self, cfg, norm_func):
        super(BaseStem, self).__init__()

        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS #64主干网络的输出通道数
        # 7*7, 64, stride = 2
        self.conv1 = Conv2d(
            3, out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_func(out_channels) #通过对应的norm_func归一化层

        for l in [self.conv1,]: #凯明初始化
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x): #定义前向传播过程
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x) # 这里stem也包括了max pool，因为无参数，直接写在forward里
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

# 当Bottleneck类实现好的时候，BottleneckWithFixedBatchNorm和BottleneckWithGN
# 就是简单的继承它就好了，然后初始化自己的参数，唯一的区别就是norm_func是FrozenBatchNorm2d
# 还是group_norm

class BottleneckWithFixedBatchNorm(Bottleneck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1
    ):
        super(BottleneckWithFixedBatchNorm, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=FrozenBatchNorm2d
        )

# 下面的StemWithFixedBatchNorm和StemWithGN继承了类BaseStem
# 只不过初始化的时候是FrozenBatchNorm2d还是group_norm
class StemWithFixedBatchNorm(BaseStem):
    def __init__(self, cfg):
        super(StemWithFixedBatchNorm, self).__init__(
            cfg, norm_func=FrozenBatchNorm2d
        )


class BottleneckWithGN(Bottleneck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1
    ):
        super(BottleneckWithGN, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=group_norm
        )


class StemWithGN(BaseStem):
    def __init__(self, cfg):
        super(StemWithGN, self).__init__(cfg, norm_func=group_norm)

# 这个指定resnet的Bottleneck结构用FixedBatchNorm还是GroupNor
_TRANSFORMATION_MODULES = Registry({
    "BottleneckWithFixedBatchNorm": BottleneckWithFixedBatchNorm,
    "BottleneckWithGN": BottleneckWithGN,
})
# 这个指定resnet的Stem结构用FixedBatchNorm还是GroupNorm
_STEM_MODULES = Registry({
    "StemWithFixedBatchNorm": StemWithFixedBatchNorm,
    "StemWithGN": StemWithGN,
})
# 这个指定具体构建resnet的哪个深度的模型，并且到第几个stage
_STAGE_SPECS = Registry({
    "R-50-C4": ResNet50StagesTo4,
    "R-50-C5": ResNet50StagesTo5,
    "R-101-C4": ResNet101StagesTo4,
    "R-101-C5": ResNet101StagesTo5,
    "R-50-FPN": ResNet50FPNStagesTo5,
    "R-50-FPN-RETINANET": ResNet50FPNStagesTo5,
    "R-101-FPN": ResNet101FPNStagesTo5,
    "R-101-FPN-RETINANET": ResNet101FPNStagesTo5,
    "R-152-FPN": ResNet152FPNStagesTo5,
})
