import torch
import os
from xml.dom.minidom import parse
from Data_Augmentation import DataAugmentForObjectDetection
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as transforms

dataAug = DataAugmentForObjectDetection()

class Mark_Dataset(Dataset):
    def __init__(self, root, transforms='train'):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        self.bbox_xml = list(sorted(os.listdir(os.path.join(root, "Annotations"))))

    def __getitem__(self, idx):
        # load images and bbox
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        bbox_xml_path = os.path.join(self.root, "Annotations", self.bbox_xml[idx])
        img = cv2.imread(img_path)
        # img = Image.open(img_path).convert("RGB")

        dom = parse(bbox_xml_path)
        data = dom.documentElement
        objects = data.getElementsByTagName('object')
        boxes = []
        labels = []
        for object_ in objects:

            name = object_.getElementsByTagName('name')[0].childNodes[0].nodeValue
            labels.append(int(name))
            bndbox = object_.getElementsByTagName('bndbox')[0]
            xmin = float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
            ymin = float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
            xmax = float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
            ymax = float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
            boxes.append([xmin, ymin, xmax, ymax])

        if self.transforms == 'train':
            img, boxes = dataAug(img, boxes)
        else:
            _to_tensor = transforms.ToTensor()
            img = _to_tensor(img)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(objects),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        return img, target

    def __len__(self):
        return len(self.imgs)

