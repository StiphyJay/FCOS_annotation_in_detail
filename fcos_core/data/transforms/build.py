# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from . import transforms as T
from fcos_core.data.transforms.transforms import Normalize, Compose, Resize, RandomHorizontalFlip, ToTensor

def build_transforms(cfg, is_train=True): #限制输入尺寸的最大最小值
    if is_train:
        if cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0] == -1: #INPUT.MIN_SIZE_RANGE_TRAIN=(-1,-1)
            min_size = cfg.INPUT.MIN_SIZE_TRAIN #(800,) 训练输入图片的最小尺寸
        else:
            assert len(cfg.INPUT.MIN_SIZE_RANGE_TRAIN) == 2, \
                "MIN_SIZE_RANGE_TRAIN must have two elements (lower bound, upper bound)"
            min_size = list(range(
                cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0],
                cfg.INPUT.MIN_SIZE_RANGE_TRAIN[1] + 1
            ))
        max_size = cfg.INPUT.MAX_SIZE_TRAIN #1333　训练输入图片的最大尺寸
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0

    to_bgr255 = cfg.INPUT.TO_BGR255 #True
    #INPUT.PIXEL_MEAN=[102.9801, 115.9465, 122.7717]
    #INPUT.PIXEL_STD = [1., 1., 1.]
    #根据图像的均值和方差对图像进行归一化
    normalize_transform = Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    transform = Compose(
        [
            Resize(min_size, max_size), #对图片进行尺寸的固定
            RandomHorizontalFlip(flip_prob), #对图片进行随机水平翻转
            ToTensor(), #将图片转成张量
            normalize_transform,
        ]
    )
    return transform
