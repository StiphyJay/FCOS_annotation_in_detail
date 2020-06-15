import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator

from fcos_core.layers import Scale
from fcos_core.config import cfg

class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels): #in_channels=256*4
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            此处是fpn每层的输出通道数　都是一样的
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1 # 81-1 COCO数据集中总的分类类别为80类

        cls_tower = [] #分类层
        bbox_tower = [] #回归层
        #head部分
        for i in range(cfg.MODEL.FCOS.NUM_CONVS): #4　分类层的四层卷积
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            # cls_tower和bbox_tower都是4层的256通道的3×3的卷积层，后加一些GN和Relu
            cls_tower.append(nn.GroupNorm(32, in_channels)) #紧接着BN层
            cls_tower.append(nn.ReLU()) #ReLU

            bbox_tower.append(  #4　回归层的四层卷积
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))  #分类部分卷积网络构建　head
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))   #回归部分卷积网络构建 head
        # 　最终的分类层　最后一个分类层用卷积网络，卷积成81个通道的输出，每个通道对应的是其类别 shape:[H*W*C]
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        #　最后的边界框预测层　输出四个预测边框数值 r l t b shape:[H*W*4]
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        # shape" [H*W*1]
        #　 最后的中心度预测层 中心度预测与分类预测在一个分支，由分类部分的四个卷积层输出的特征再经过一个卷积层输出一个对应的分数
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization 这些层里面的卷积参数都进行初始化
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d): #isinstance是考虑l是不是nn.Conv2d的类型
                    torch.nn.init.normal_(l.weight, std=0.01) #初始化权重服从正态分布
                    torch.nn.init.constant_(l.bias, 0)  #初始偏置为０

        # initialize the bias for focal loss 我只知道分类是用focal loss，可能是一种经验trick?
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB #0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value) #分类的偏置初始化

        # P3-P7共有5层特征FPN，缩放因子，对回归结果进行缩放
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = [] #分类
        bbox_reg = [] #预测
        centerness = [] #中心度
        # 我想这里的x应该是fpn出来的各层特征，因为x根据下一句看是可迭代的
        for l, feature in enumerate(x):
            # 要注意，不图层经过tower之后的特征图大小是不一样的
            # 还有一点就是，不同层的特征都是共享一个tower，无论是cls分支还是bbox分支
            cls_tower = self.cls_tower(feature) #经过特征金字塔提取的每一个层次的特征
            logits.append(self.cls_logits(cls_tower)) #对应的分类得分

            centerness.append(self.centerness(cls_tower)) #对应的中心度数值
            bbox_reg.append(torch.exp(self.scales[l](
                self.bbox_pred(self.bbox_tower(feature))  #回归盒子的得分
            )))
        return logits, bbox_reg, centerness

class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels): #cfg in_channels=256*4
        super(FCOSModule, self).__init__()
        #对应的FCOS　head部分的网络
        head = FCOSHead(cfg, in_channels) #得到head　每个head返回特征对应的框体分类结果，框体回归结果以及对应的中心度计算结果

        box_selector_test = make_fcos_postprocessor(cfg) #对得到的候选框进行NMS做后处理

        loss_evaluator = make_fcos_loss_evaluator(cfg) #计算loss
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES #[8, 16, 32, 64, 128] 五个不同层的特征金字塔的跨度stride

    def forward(self, images, features, targets=None): # 调用的时候:self.rpn(images, features, targets)
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.　一个图像中返回一个框体列表
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        #以P3[256,100,128]为例，其得到的box_cls[80,100,128] box_regression[4,100,128] centerness[1,100,128]　计算出来的location[12800,2]
        box_cls, box_regression, centerness = self.head(features) #获得预测的分类、回归以及中心度数值
        locations = self.compute_locations(features) #根据特征计算其在原始输入图片上的对应的位置
        # targets是对应的真值　location是当前的特征对应的原始输入图片的像素位置
        # box_cls　box_regression centerness　是该原始输入图片的像素对应预测的框体分类结果，框体回归结果已及中心度
        if self.training:
            return self._forward_train(
                locations, box_cls, 
                box_regression, 
                centerness, targets
            )

        #测试时不需要真值
        else:
            return self._forward_test(
                locations, box_cls, box_regression, 
                centerness, images.image_sizes
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness, targets):
        #分类，回归，中心度损失
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, targets
        )
        #得到对应的损失字典
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression, 
            centerness, image_sizes
        )
        return boxes, {}
    #根据提取的高维特征　该特征包含不同的金字塔维度　以及相应维度对应的特征
    def compute_locations(self, features): #features=B*256*100*128对应的是P3
        locations = []
        for level, feature in enumerate(features): #level是不同的级别P3-P7
            h, w = feature.size()[-2:]
            #self.fpn_strides = [8, 16, 32, 64, 128]
            #以P3为例 h=100 w=128 s=8 locations_per_level的尺寸为[12800,2],内容为[[4,4],[12,4],...[1020,4]...[1020,796]]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level], #level对应的是特征的级别0 1 2 3 4 h,w对应的是该级别的特征图的长和宽
                feature.device
            )
            locations.append(locations_per_level) #location对应的每一层的特征图中的对应位置 locations这个列表一共有五个元素
        return locations
#计算特征金字塔中每一层的特征图中每个像素对应到原始输入图像中的位置，该对应位置一般在感受野中心附近.
    def compute_locations_per_level(self, h, w, stride, device):
        #对特征图的每个像素在x,y上都进行对应步长的偏移
        shifts_x = torch.arange(
            0, w * stride, step=stride, #步长
            dtype=torch.float32, device=device
        )#以8为间隔，一个列表，从0-1016　即[0,8,16,...,1016]
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )#以8为间隔，一个列表，从0-792　即[0,8,16,...,792]
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x) #组成网格
        shift_x = shift_x.reshape(-1) #
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2 #向下取证　使得原图上的对应点尽可能接近location(x,y)的感受野中心
        return locations #

def build_fcos(cfg, in_channels):
    return FCOSModule(cfg, in_channels)


if __name__ == '__main__':
    in_channels = torch.randn(256, 4).cuda() #batch_size=2
    # search_data = torch.randn(256,4).cuda()
    #
    net = FCOSHead().cuda()
    #
    # # gpu_tracker.track()
    # # startT = time.process_time()
    rpn_output = net(cfg, in_channels)
    # overT = time.process_time()
    # gpu_tracker.track()
    # print('time: ', overT-startT)
    print('cls_feature: ', rpn_output['cls_feature'].shape)
    print('reg_feature: ', rpn_output['reg_feature'].shape)
