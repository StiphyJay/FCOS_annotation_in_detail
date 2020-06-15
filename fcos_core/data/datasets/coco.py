# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.segmentation_mask import SegmentationMask
from fcos_core.structures.keypoint import PersonKeypoints


min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False

class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids) #对于val2017文件夹下面的图片　按照他们的命名编号进行排序

        # filter images without detection annotations
        if remove_images_without_annotations: #True
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids
        ## category_id为coco类别id,json_category_id_to_contiguous_id 为字典类型
        # 其中, key为coco的非连续id, value为1~80的连续id, 均为整数, 所以这里是将coco的非连续id转换成对应的连续id
        #self.coco.getCatIds() 1-80对应着80个类
        #json_category_id_to_contiguous_id = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
        # 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
        # 18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23,
        # 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30,
        # 35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37,
        # 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44,
        # 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51,
        # 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58,
        # 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65,
        # 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
        # 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0] #将label文件转成列表

        boxes = [obj["bbox"] for obj in anno] #读取图片中物体的真值框数据[x,y,w,h]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes 生成一个张量 n*4 4这个维度表示的是x,y,w,h, n表示的是　一张图片里有多少个真值框
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy") #按照对应的类　初始化一下

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes) #target.extra_fields["labels"] = classes

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size, mode='poly')
        target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def get_img_info(self, index): #从图片id找到对应的图片信息
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

if __name__ == "__main__":
    root = '/home/sifan/slam-package/FCOS/datasets/coco/val2017/'
    annFile = '/home/sifan/slam-package/FCOS/datasets/coco/annotations/instances_val2017.json'
    #C = COCODataset(annFile, root, True)
    #print(C.json_category_id_to_contiguous_id)