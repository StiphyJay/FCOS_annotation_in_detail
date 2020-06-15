# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy
import logging
import sys
sys.path.append('..')
import torch.utils.data
from fcos_core.utils.comm import get_world_size
from fcos_core.utils.imports import import_file
from fcos_core.data.datasets import evaluation as e
# from fcos_core.data import datasets as D
from . import datasets as D
from . import samplers

from .collate_batch import BatchCollator
# from .transforms import build_transforms

from fcos_core.data.transforms.build import build_transforms

def build_dataset(dataset_list, transforms, dataset_catalog, is_train=True):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_trian, coco_2014_val, etc 数据集列表
        transforms (callable): transforms to apply to each (image, target) sample　对每个图片样本进行变换处理
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list: # coco_2014_trian, coco_2014_val
        #data 中的一个例子
        #{'factory': 'COCODataset', 'args': {'root': 'datasets/coco/val2014',
        # 'ann_file': 'datasets/coco/annotations/instances_val2014.json'}}
        data = dataset_catalog.get(dataset_name) #直接输入数据集文件路径，返回对应的数据集，以及对应的数据集的图片路径和真值路径
        #getattr函数返回对象的属性值  data["factory"]=COCODataset
        #此处用于返回函数 也就是返回fcos_core/data/datasets/coco.py中的 COCODataset
        factory = getattr(D, data["factory"]) #getattr函数返回对象的属性值
        args = data["args"] #对应的数据集图片路径和标注的路径以字典的形式保存在args中
        #args: {'root': 'datasets/coco/val2014',
        #'ann_file': 'datasets/coco/annotations/instances_val2014.json'}

        # for COCODataset, we want to remove images without annotations
        # during training
        if data["factory"] == "COCODataset":
            args["remove_images_without_annotations"] = is_train
        if data["factory"] == "PascalVOCDataset":
            args["use_difficult"] = not is_train
        args["transforms"] = transforms
        # make dataset from factory
        dataset = factory(**args) #**args用于将字典中的值取出　然后实例化factory 即上述的COCOdataset函数
        #此处的dataset到底是什么形式？
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)): #if False
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler

# during train:   is_train=True, is_distributed=distributed, False   start_iter=arguments["iteration"] =0
def make_data_loader(cfg, is_train=True, is_distributed=False, start_iter=0):
    num_gpus = get_world_size() #判断gpu数量
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH #16 每个batch_size的图片是16张
        assert (
            images_per_batch % num_gpus == 0  #判断每个batch_size的图片可以均匀的分到多个gpu上面
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER #40000 最大迭代次数不超过40000

    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    if images_per_gpu > 1: #提示关于训练过程中的内存不足的问题
        logger = logging.getLogger(__name__)
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
        )
    #将图片进行分组，仅仅根据两种情形分组，一种是图片的宽/高>1的，一种是其他的．
    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else [] #True

    #PATHS_CATALOG=os.path.join(os.path.dirname(__file__), "paths_catalog.py")
    #找出对应的加载数据集脚本的路径
    paths_catalog = import_file(
        "fcos_core.config.paths_catalog", cfg.PATHS_CATALOG, True)
    #DatasetCatalog 对应的是fcos_core.config.paths_catalog中的DatasetCatalog类，并对其进行实例化.
    DatasetCatalog = paths_catalog.DatasetCatalog #对应要训练的数据集路径 <class 'fcos_core.config.paths_catalog.DatasetCatalog'>
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST #数据集列表　训练或是测试　对应的列表中的数据集不一样
    # train: ("coco_2014_train", "coco_2014_valminusminival")
    # test:  ("coco_2014_minival",)
    print(dataset_list)

    transforms = build_transforms(cfg, is_train) #对输入图片进行变换，随机水平分割归一化等操作
    datasets = build_dataset(dataset_list, transforms, DatasetCatalog, is_train)

    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter)

        collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders


import os
from fcos_core.config import defaults as cfg
if __name__ == "__main__":
    PATHS_CATALOG="/home/sifan/slam-package/FCOS/fcos_core/config/paths_catalog.py"
    paths_catalog = import_file("fcos_core.config.paths_catalog", PATHS_CATALOG, True)
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = () if True else ()
    is_train= True
    transforms = build_transforms(cfg, is_train)
    data = {'factory': 'COCODataset', 'args': {'root': 'datasets/coco/train2014', 'ann_file': 'datasets/coco/annotations/instances_train2014.json'}}
    #factory = getattr(D, data["factory"])
    #print(factory)