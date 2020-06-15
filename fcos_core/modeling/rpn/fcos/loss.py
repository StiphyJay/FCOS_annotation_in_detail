"""
This file contains specific functions for computing losses of FCOS
file
"""

import torch
from torch.nn import functional as F
from torch import nn

from ..utils import concat_box_prediction_layers
from fcos_core.layers import IOULoss
from fcos_core.layers import SigmoidFocalLoss
from fcos_core.modeling.matcher import Matcher
from fcos_core.modeling.utils import cat
from fcos_core.structures.boxlist_ops import boxlist_iou
from fcos_core.structures.boxlist_ops import cat_boxlist


INF = 100000000

 #locations, box_cls, box_regression, centerness, targets
class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,  #2.0
            cfg.MODEL.FCOS.LOSS_ALPHA   #0.25
        )
        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss() #IOU损失
        self.centerness_loss_func = nn.BCEWithLogitsLoss() #对中心度部分做二元交叉损失

    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        #expanded_object_sizes_of_interest按照每个level创建了该level所有采样点的sizes_of_interest，
        #然后用torch.cat合并起来，形成了(N, 2)形状的数据，N为所有采样点的个数
        #以P3为例，l为点的序数，points_per_level为对应的像素坐标
        #设置每个location的感兴趣尺寸大小，P3里面的每个location的size_of_interest 是[-1,64]
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )
        #对于每一个采样点，都需要有一个对应的sizes_of_interest。expanded_object_sizes_of_interest按照
        # 每个level创建了该level所有采样点的sizes_of_interest，然后用torch.cat合并起来，形成了(N, 2)形状的
        # 数据，N为所有采样点的个数
        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        #num_points_per_level=[12800, 3200, 800, 208, 56]
        num_points_per_level = [len(points_per_level) for points_per_level in points] #每个level的点个数　用于后续的操作
        points_all_level = torch.cat(points, dim=0) #包含了所有采样点，跟expanded_object_sizes_of_interest类似，
        # 也用torch.cat合并，形成了(N, 2)的形状。

        #该函数的作用是使用points_all_level, targets, expanded_object_sizes_of_interest计算
        # 分类和回归的标注即labels和reg_targets，形状为(P, N)和(P, N, 4)，P为图像数量。
        #一张图像时，对应的labels是N，reg_targets是N*4
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )
        #compute_targets_for_locations函数得到的labels和reg_targets是把所有level的数据拼接在一起的。
        # 现在要根据每个level的点个数即num_points_per_level把他们拆开，按照level优先做成一个list。
        # 最后的结果是labels_level_first[level]，拥有P*N_level个元素。reg_targets_level_first[level]形状
        # 为(P*N_level, 4)，N_level为该level采样点的个数
        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )
            reg_targets_level_first.append(
                torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
            )

        return labels_level_first, reg_targets_level_first


#由于每张图像可能含有不同数量的bbox，所以先读取targets的labels和bbox，
# 创建labels_per_im变量和reg_targets_per_im变量（形状为(N, M, 4)，N为所有采样点的个数，M为bbox数量）

#根据论文3.2节，不同尺度大小的bbox将被分配到不同的fpn level去计算，不在level对应范围的bbox将被忽略。
# 这样操作之后若某一个采样点仍然对应到多个bbox，则取最小面积的bbox。fcos中的实现是把level对应范围之外
# 的bbox面积设成无穷大，用locations_to_gt_area这个变量实现位置到gt中面积的映射，代码48行使reg_targets_per_im
# 取到面积最小的bbox。同理每个采样点也有一个对应的类别，对于不在gt bbox的点类别设成背景。最终的labels_per_im形状为(N)，
# reg_targets_per_im形状为(N, 4)。
#compute_targets_for_locations函数使用points_all_level, targets, expanded_object_sizes_of_interest
# 计算分类和回归的标注即labels和reg_targets，形状为(P, N)和(P, N, 4)，P为图像数量
    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i] #第i个图像的target
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox #图像的边界框信息　左上角坐标和右下角坐标
            labels_per_im = targets_per_im.get_field("labels") #得到边框中图像的对应类别
            area = targets_per_im.area() #计算出边界框的像素面积
            #这里的x,y表示的是每一层的所有的采样点聚合在一起 也就是[4,4] [12,4] [20,4] ... [960,832]
            l = xs[:, None] - bboxes[:, 0][None] #l=x-x0  x0,y0为边界框的左上角坐标
            t = ys[:, None] - bboxes[:, 1][None] #t=y-y0
            r = bboxes[:, 2][None] - xs[:, None] #r=x1-x
            b = bboxes[:, 3][None] - ys[:, None] #b=y1-y  x1,y1为边界框的右下角坐标
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)#每个采样点的坐标减去边界框的左上角和右下角坐标获得的差值　
            # N*M*4 N为采样点个数 M为该张图片中bbox数量　4是回归目标 17064*1*4
            #找出每一个采样点与边界框做差值后里面的l,t,r,b中的最小值，如果最小值也大于０，说明该采样点在边界框里
            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0 #判断哪个采样点是在目标里面　

            #同理，找出每一个采样点与边界框做差值后里面的l,t,r,b中的最大值，来限制每一层的回归范围
            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location P3级别的location和边界框真值做完差值l* t* r* b*之后，
            # 找出其中的的最大值，在P3级别中该最大值得在0-64之间，超过64像素值的location部分将其作为负标签，不再回归，
            # 同理P4部分的回归最大距离为64-128，以此在不同特征level中，回归不同尺寸的边界框．
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])
            #0< max_reg_targets<64 会被考虑
            locations_to_gt_area = area[None].repeat(len(locations), 1) #面积复制17064份 17064*1
            locations_to_gt_area[is_in_boxes == 0] = INF #不在边界框里面的边界框区域都给无穷大
            locations_to_gt_area[is_cared_in_the_level == 0] = INF #不在对应的特征层范围内的　　回归的边界框面积也都给无穷大

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            #获得里面的最小面积以及对应的列表序号
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)
            #选择面积最小的bbox
            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds] #对应的location　ID　都给真值label
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets
    #中心度计算
    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        #返回分类，回归和中心度损失
        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """

        N = box_cls[0].size(0) #应该是图片数量
        num_classes = box_cls[0].size(1) #分类的类别数量
        labels, reg_targets = self.prepare_targets(locations, targets) #根据特征层的位置计算出最初输入图片上对应的框的标签和位置

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for l in range(len(labels)):
            #预测的边界框分类
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes)) #12800*80
            #预测的边界框回归
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            #边界框分类label 读取真值，将location与真值计算获得
            labels_flatten.append(labels[l].reshape(-1))
            #边界框回归label　读取真值，将location与真值计算获得
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            #预测的中心度
            centerness_flatten.append(centerness[l].reshape(-1))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        #分类loss计算 box_cls_flatten预测的分类结果 labels_flatten对去真值标签计算得到的
        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero


        box_regression_flatten = box_regression_flatten[pos_inds] #预测的边界框回归
        reg_targets_flatten = reg_targets_flatten[pos_inds] #边界框回归label　读取真值，将location与真值计算获得
        centerness_flatten = centerness_flatten[pos_inds]  #预测的中心度

        if pos_inds.numel() > 0: #numel()返回元素数目
            #用location与真值构建的reg来构建中心度的label
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            #回归部分损失 输入为预测的边界框　边界框的lable　以及中心度的label
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss

#locations, box_cls, box_regression, centerness, targets
def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator
