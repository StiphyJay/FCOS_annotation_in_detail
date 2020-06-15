# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from fcos_core.structures.image_list import to_image_list

from fcos_core.modeling.backbone import build_backbone
from fcos_core.modeling.rpn.rpn import build_rpn
from fcos_core.modeling.roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """
    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        #self.backbone返回的是一个nn.Sequential的model,其中FPN出来的就是那多层特征
        self.backbone = build_backbone(cfg) #backbone　基础网络　提取特征 网络输出为out_channels = 256*4
        #这里构造的是fcos_head
        self.rpn = build_rpn(cfg, self.backbone.out_channels) #生成提议的box以及对应的分类、位置回归以及中心度 对应fcos中的head
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels) #

    def forward(self, images, targets=None): #传入图片和对应真值
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images) #传入训练图片
        features = self.backbone(images.tensors) #基础网络提取的特征 #B*1024*50*64
        #测试时候会返回proposal　形成检测框　训练时候只返回loss数值
        proposals, proposal_losses = self.rpn(images, features, targets)  #提取的特征经过RPN生成提议框
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result

from fcos_core.config import cfg
if __name__=='__main__':
    img = torch.randn(3, 800, 1024)
    backbone = build_backbone(cfg)
    features = backbone(img)
    print(features)
    #rpn = build_rpn(cfg, 256*4)
    #print(rpn)
#下面是打印出来的backbone网络
'''
input= (3*800*1024) 经过stem　->(64,200,256) 经过layer1 -> 256*200*256 
                                            经过layer2 -> 512*100*128 对应论文中的C3
                                            经过layer3 -> 1024*50*64  对应论文中的C4
                                            经过layer4 -> 2048*25*32  对应论文中的C5

Sequential(
  (body): ResNet(
    (stem): StemWithFixedBatchNorm(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): FrozenBatchNorm2d())
    (layer1): Sequential(
      (0): BottleneckWithFixedBatchNorm(
        (downsample): Sequential(
          (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): FrozenBatchNorm2d()
        )
        (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
      (1): BottleneckWithFixedBatchNorm(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
      (2): BottleneckWithFixedBatchNorm(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
    )
    (layer2): Sequential(
      (0): BottleneckWithFixedBatchNorm(
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): FrozenBatchNorm2d()
        )
        (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
      (1): BottleneckWithFixedBatchNorm(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
      (2): BottleneckWithFixedBatchNorm(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
      (3): BottleneckWithFixedBatchNorm(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
    )
    (layer3): Sequential(
      (0): BottleneckWithFixedBatchNorm(
        (downsample): Sequential(
          (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): FrozenBatchNorm2d()
        )
        (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
      (1): BottleneckWithFixedBatchNorm(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
      (2): BottleneckWithFixedBatchNorm(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
      (3): BottleneckWithFixedBatchNorm(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
      (4): BottleneckWithFixedBatchNorm(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
      (5): BottleneckWithFixedBatchNorm(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): FrozenBatchNorm2d()
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): FrozenBatchNorm2d()
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): FrozenBatchNorm2d()
      )
    )
    (layer4): Sequential(
        (0): BottleneckWithFixedBatchNorm(
          (downsample): Sequential(
            (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): FrozenBatchNorm2d()
          )
          (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (bn1): FrozenBatchNorm2d()
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d()
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d()
        )
        (1): BottleneckWithFixedBatchNorm(
          (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d()
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d()
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d()
        )
        (2): BottleneckWithFixedBatchNorm(
          (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d()
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d()
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d()
        )
      )
    )
)
'''

'''
#对应论文中的P5-P4-P3 
C5(2048*25*32)->fpn_inner4(256*25*32)->fpn_layer4(256*25*32)=P5
C4(1024*50*64)->{fpn_inner3(256*50*64)+P5(256*25*32)->inner_top_down(256*50*64)}->fpn_layer3(256*50*64)=P4
C3(512*100*128)->{fpn_inner2(256*100*128)+P4(256*50*64)->inner_top_down(256*100*128)}->fpn_layer2(256*100*128)=P3
P6 P7分别在由P5和P6下采样得到

(fpn): FPN(
      (fpn_inner2): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      (fpn_layer2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (fpn_inner3): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
      (fpn_layer3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (fpn_inner4): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
      (fpn_layer4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (top_blocks): LastLevelP6P7(
        (p6): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (p7): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      )
    )
  )
  以p3为例子 P3(256*100*128)->cls_tower(256*100*128)->cls_logits(80*100*128) 80个类别
            P3(256*100*128)->cls_tower(256*100*128)->centerness(1*100*128) centerness
            P3(256*100*128)->bbox_tower(256*100*128)->bbox_pred(4*100*128) 边界框的４个数值
            
  (rpn): FCOSModule(
    (head): FCOSHead(
      (cls_tower): Sequential(
        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): GroupNorm(32, 256, eps=1e-05, affine=True)
        (2): ReLU()
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): GroupNorm(32, 256, eps=1e-05, affine=True)
        (5): ReLU()
        (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): GroupNorm(32, 256, eps=1e-05, affine=True)
        (8): ReLU()
        (9): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (10): GroupNorm(32, 256, eps=1e-05, affine=True)
        (11): ReLU()
      )
      (bbox_tower): Sequential(
        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): GroupNorm(32, 256, eps=1e-05, affine=True)
        (2): ReLU()
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): GroupNorm(32, 256, eps=1e-05, affine=True)
        (5): ReLU()
        (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): GroupNorm(32, 256, eps=1e-05, affine=True)
        (8): ReLU()
        (9): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (10): GroupNorm(32, 256, eps=1e-05, affine=True)
        (11): ReLU()
      )
      (cls_logits): Conv2d(256, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bbox_pred): Conv2d(256, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (centerness): Conv2d(256, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (scales): ModuleList(
        (0): Scale()
        (1): Scale()
        (2): Scale()
        (3): Scale()
        (4): Scale()
      )
    )
    (box_selector_test): FCOSPostProcessor()
  )
'''