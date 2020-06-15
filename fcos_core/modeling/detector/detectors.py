# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# import fcos_core.modeling.detector.generalized_rcnn as g
from .generalized_rcnn import GeneralizedRCNN
# from fcos_core.config import cfg

_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES["GeneralizedRCNN"]  #_cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    return meta_arch(cfg) #返回GeneralizedRCNN(cfg)

if __name__ =="__main__":
    print(build_detection_model())
