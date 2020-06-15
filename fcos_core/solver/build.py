# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR #学习率0.001
        weight_decay = cfg.SOLVER.WEIGHT_DECAY #0.0005
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM) # 定义一个随机梯度下降的优化器momentum = 0.9
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        #SOLVER.GAMMA = 0.1
        #SOLVER.STEPS = (30000,)
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        #SOLVER.WARMUP_FACTOR = 1.0 / 3
        #SOLVER.WARMUP_ITERS = 500
        #SOLVER.WARMUP_METHOD = "linear"
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
