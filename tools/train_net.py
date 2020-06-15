#!usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
#from fcos_core.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import sys
print(__file__)
curPath = os.path.abspath(os.path.dirname(__file__))
print(curPath)  #/home/sifan/slam-package/FCOS/tools
rootPath = os.path.split(curPath)[0]
print(rootPath) #/home/sifan/slam-package/FCOS
sys.path.append(rootPath)
import torch
from fcos_core.config import cfg
from fcos_core.data import make_data_loader
from fcos_core.solver import make_lr_scheduler
from fcos_core.solver import make_optimizer
from fcos_core.engine.inference import inference
from fcos_core.engine.trainer import do_train
from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.utils.collect_env import collect_env_info
from fcos_core.utils.comm import synchronize, \
    get_rank, is_pytorch_1_1_0_or_later
from fcos_core.utils.imports import import_file
from fcos_core.utils.logger import setup_logger
from fcos_core.utils.miscellaneous import mkdir

def train(cfg, local_rank, distributed): #cfg 0 False
    model = build_detection_model(cfg) #实例化模型
    device = torch.device(cfg.MODEL.DEVICE) #cfg.MODEL.DEVICE="cuda" 将torch.tensor分配到cuda　即GPU上
    model.to(device) #将模型放在gpu上运行

    if cfg.MODEL.USE_SYNCBN: #False
        assert is_pytorch_1_1_0_or_later(), \
            "SyncBatchNorm is only available in pytorch >= 1.1.0"
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = make_optimizer(cfg, model) # 定义网络训练优化器
    scheduler = make_lr_scheduler(cfg, optimizer)  #设置学习率

    if distributed: #False
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR #"."

    save_to_disk = get_rank() == 0 #True
    #checkpoint为网络的预训练模型
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data) #将extra_checkpoint_data字典里的数值加入到arguments字典中

    #make_data_loader
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed, #False
        start_iter=arguments["iteration"], # 0
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD  #SOLVER.CHECKPOINT_PERIOD = 2500

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
    )

    return model


def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()

def main():
    # 解析命令行参数，例如--config-file
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file", #配置文件
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    #此参数是通过torch.distributed.launch传递过来的，我们设置位置参数来接受
    # local_rank代表当前程序进程使用的GPU标号
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER, #所有剩余的命令行参数都被收集到一个列表中
    )

    args = parser.parse_args()
    #判断机器上gpu的数量，大于1时自动使用分布式训练
    #world_size是由torch.distributed.launch.py产生
    # 具体数值为 nproc_per_node*node(node就是主机数)
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1 #判断当前系统环境变量中是否有"WORLD_SIZE"　如果没有num_gpus=1
    args.distributed = num_gpus > 1 #False

    if args.distributed: #False
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group\
        (
            backend="nccl", init_method="env://"
        )
        synchronize()
    #yacs的具体用法　可以参考印象笔记
    #参数默认在fcos_core/config_defaults.py中 其余参数由config_file opts覆盖
    cfg.merge_from_file(args.config_file) #从yaml文件中读取参数 即configs/fcos/fcos_R_50_FPN_1x.yaml
    cfg.merge_from_list(args.opts) #也可以从命令行进行参数重写
    cfg.freeze() #冻结参数　防止不小心被更改 cfg被传入train()

    output_dir = cfg.OUTPUT_DIR #输出模型路径　存放一些日志信息
    if output_dir:
        mkdir(output_dir) #创建对应的输出路径

    #写入日志文件　包括gpu数量，系统环境，配置文件参数等
    logger = setup_logger("fcos_core", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args.local_rank, args.distributed) #local_rank=0 distributed=False

    if not args.skip_test:
        run_test(cfg, model, args.distributed)


if __name__ == "__main__":
    #overall_network = build_detection_model(cfg)
    #print(overall_network)
    main()
