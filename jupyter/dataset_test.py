import torch
import torchvision
import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision.transforms import functional as F
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import random
from PIL import Image


font = cv2.FONT_HERSHEY_SIMPLEX

root = '/home/sifan/FCOS/datasets/coco/val2017/'
annFile = '/home/sifan/FCOS/datasets/coco/annotations/instances_val2017.json'

# 定义 coco collate_fn
def collate_fn_coco(batch):
    return tuple(zip(*batch))
def coco_test(root, annFile):
    # 创建 coco dataset
    coco_det = datasets.CocoDetection(root, annFile,transform=T.ToTensor())
    #coco_det = datasets.CocoDetection(root, annFile)
    # 创建 Coco sampler
    sampler = torch.utils.data.RandomSampler(coco_det)
    batch_sampler = torch.utils.data.BatchSampler(sampler, 8, drop_last=True)

    # 创建 dataloader
    data_loader = torch.utils.data.DataLoader(
        coco_det, batch_sampler=batch_sampler, num_workers=3,
        collate_fn=collate_fn_coco)

# 可视化 一张图片对应的真值数据　该真值数据是从dataset.cocodetection中反馈回来的．接下来要解析该真值数据.
#[{'segmentation': [[24.89, 117.38, 130.21, 126.95, 206.8, 132.7, 256.58, 140.35, 289.13, 145.14,
# 335.09, 156.63, 386.79, 166.2, 423.17, 178.65, 433.7, 176.74, 480.61, 187.27, 518.91, 193.97, 550.5,
# 203.54, 576.35, 218.86, 579.23, 227.48, 576.35, 274.39, 573.48, 285.88, 579.23, 286.84, 576.35, 301.2,
# 564.87, 314.6, 533.27, 314.6, 507.42, 312.69, 401.15, 249.5, 324.56, 217.9, 254.67, 181.52, 154.14,
# 148.01, 127.33, 144.18, 108.19, 138.44, 75.63, 140.35, 45.96, 128.87, 19.15, 119.29]], #RLE or [polygon],# 对象的边界点（边界多边形，此时iscrowd=0）。
#segmentation格式取决于这个实例是一个单个的对象（即iscrowd=0，将使用polygons格式）还是一组对象（即iscrowd=1，将使用RLE格式）
#也就是说当前的边界框里面存不存在覆盖的现象，比如人手里拿一个棒球手套这种情况.
# 'area': 32509.091900000014, #区域面积 float
# 'iscrowd': 0,
# 'image_id': 130566, #图像的命名　此图片是数据集文件夹下的 000000130566.jpg
# 'bbox': [19.15, 117.38, 560.08, 197.22], #定位边框[x,y,w,h]
# 'category_id': 7, 类别ID 对应到整个数据集可以的分类的类别中
# 'id': 172896}] 对象ID　因为每张图像里面不只有一个ID　所以要对每一个对象编号(每个对象的ID是唯一的)
#对应图片id为130566 读取完图片信息和真值数据　将真值边框在输入图片上显示
    for imgs, labels, paths in data_loader:
        print(len(imgs))
        print(len(labels))
        print(len(paths))
        for i in range(len(imgs)):
            bboxes = []
            ids = []
            img = imgs[i]
            labels_ = labels[i]
            path = paths[i]
            print(img)
            print(labels_)
            print(path)
            #cv2.imshow('test', img.numpy())
            #cv2.waitKey()
            for label in labels_: #labels_是一个列表，该列表中的元素是一个个的字典，每个字典表示的是图像中一个物体的类别中的相关真值信息
                print(label)
                bboxes.append([label['bbox'][0], # 目标物体的框左上角的横坐标x数值
                           label['bbox'][1], # 目标物体的框左上角的纵坐标y数值
                           label['bbox'][0] + label['bbox'][2], #　x + w
                           label['bbox'][1] + label['bbox'][3] # y + h
                           ])#bboxes中存放的一副图像中所有的bbox的坐标，x, y, x+w, y+h
                ids.append(label['category_id']) #ids中存放的是一幅图像中所有的物体的类别id

            img = img.permute(1,2,0).numpy() #转换一下图像维度的位置
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) #颜色空间转换函数 将RGB图片转换成BGR格式
            for box ,id_ in zip(bboxes,ids):
                x1 = int(box[0])#x1,y1 bbox的左上角坐标
                y1 = int(box[1])
                x2 = int(box[2])#x2,y2 bbox的右下角坐标
                y2 = int(box[3])
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),thickness=2) #在图像上画举行　输入左上角和右下角坐标
                cv2.putText(img, text=str(id_), org=(x1 + 5, y1 + 5), fontFace=font, fontScale=1,
                        thickness=2, lineType=cv2.LINE_AA, color=(0, 255, 0)) # 该函数在图片中添加文字 参数依次表示:图片，添加的文字，左上角坐标，字体，字体大小，字体粗细,颜色，
            cv2.imshow('test',img)
            cv2.waitKey()

from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.segmentation_mask import SegmentationMask
from fcos_core.config import cfg
from fcos_core.data.transforms.build import build_transforms
from fcos_core.modeling.detector import build_detection_model
if __name__ == "__main__":
    imgpath='/home/sifan/FCOS/datasets/coco/val2017/000000130566.jpg'
    img = Image.open(imgpath).convert('RGB') #上面程序中的img
    #上面程序中的label
    target = [{'segmentation': [[24.89, 117.38, 130.21, 126.95, 206.8, 132.7, 256.58, 140.35, 289.13,
                                 145.14, 335.09, 156.63, 386.79, 166.2, 423.17, 178.65, 433.7, 176.74,
                                 480.61, 187.27, 518.91, 193.97, 550.5, 203.54, 576.35, 218.86, 579.23,
                                 227.48, 576.35, 274.39, 573.48, 285.88, 579.23, 286.84, 576.35, 301.2,
                                 564.87, 314.6, 533.27, 314.6, 507.42, 312.69, 401.15, 249.5, 324.56,
                                 217.9, 254.67, 181.52, 154.14, 148.01, 127.33, 144.18, 108.19, 138.44,
                                 75.63, 140.35, 45.96, 128.87, 19.15, 119.29]],
               'area': 32509.091900000014,
               'iscrowd': 0,
               'image_id': 130566,
               'bbox': [19.15, 117.38, 560.08, 197.22],
               'category_id': 7,
               'id': 172896}]

    anno = target
    anno = [obj for obj in anno if obj["iscrowd"] == 0]

    boxes = [obj["bbox"] for obj in anno]
    print(boxes)
    boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
    print('boxes', boxes)
    print('img.size', img.size)
    target = BoxList(boxes, img.size, mode="xywh").convert("xyxy") #按照对应的类　初始化一下
    print(target.bbox)

    classes = [obj["category_id"] for obj in anno]
    print('classes', classes)
    # from pycocotools.coco import COCO
    # coco = COCO(annFile)
    # json_category_id_to_contiguous_id = {
    #     v: i + 1 for i, v in enumerate(coco.getCatIds())}
    # classes = [json_category_id_to_contiguous_id[c] for c in classes]
    # print('classes', classes)
    classes = 7
    classes = torch.tensor(classes)
    target.add_field("labels", classes) #self.extra_fields["labels"] = 7

    masks = [obj["segmentation"] for obj in anno]
    print('masks', masks)
    masks = SegmentationMask(masks, img.size, mode='poly')
    print('masks', masks)
    target.add_field("masks", masks) #self.extra_fields["masks"] = SegmentationMask(num_instances=1, image_width=640, image_height=427, mode=poly)
    print(target)
    print(target.bbox)
    #target = target.clip_to_image(remove_empty=True)
    transforms = build_transforms(cfg, True)
    img, target = transforms(img, target)
    print(img.shape)
    print(target)
    print(target.bbox)
    #model = build_detection_model(cfg)
    #loss_dict = model(img, target)
    #print(loss_dict)

    labels_per_im = target.get_field("labels")
    print('labels_per_im',labels_per_im)
    area = target.area()
    print(area)