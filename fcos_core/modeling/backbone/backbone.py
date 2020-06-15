# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn

from fcos_core.modeling import registry
from fcos_core.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import resnet
from . import mobilenet

@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS #256 * 4
    return model

@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model

#yaml文件中的conv_body是对应的R-50-FPN-RETINANET
#build_resnet_fpn_p3p7_backbone=registry.BACKBONES.register("R-50-FPN-RETINANET")(build_resnet_fpn_p3p7_backbone)
@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):#构建论文中backbone部分以及特征金字塔的P3-P7部分网络
    body = resnet.ResNet(cfg) #构建body backbone中的C3-C5
    #获取fpn所需要的channels参数
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS #256
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS #256*4
    #cfg.MODEL.RETINANET.USE_C5 = True
    # in_channels_p6p7 = 256*8
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    #构建fpn部分网络
    fpn = fpn_module.FPN(
        in_channels_list=[
            0, # 因为从C3起才有P3，所以stage2跳过，设置为0
            in_channels_stage2 * 2, #c3
            in_channels_stage2 * 4, #c4
            in_channels_stage2 * 8, #c5
        ],
        out_channels=out_channels, #256*4
        conv_block=conv_with_kaiming_uniform( # 这个conv如果stride=1的话就不变size,返回的是nn.Conv2d
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU #false false
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels), #256 256
    )
    # 通过有序字典将body和fpn送入nn.Sequential构造模型
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)])) # 写成一个,body的输出作为fpn的输入
    # 这个是为了之后有用，再赋一次值
    model.out_channels = out_channels #256*4
    return model


@registry.BACKBONES.register("MNV2-FPN-RETINANET")
def build_mnv2_fpn_backbone(cfg):
    body = mobilenet.MobileNetV2(cfg)
    in_channels_stage2 = body.return_features_num_channels
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2[1],
            in_channels_stage2[2],
            in_channels_stage2[3],
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(out_channels, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


def build_backbone(cfg):
    #如果CONV_BODY不在registry.BACKBONES中就抛出异常
    #MODEL.BACKBONE.CONV_BODY = "R-50-C4"
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg) #用到了装饰器的语法
    # registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY] ==> 指代build_resnet_fpn_p3p7_backbone()
    # 所以后面加一个参数:cfg