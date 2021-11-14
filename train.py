import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logger
from torchvision import models
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict
import os
from train_val_data_split import dataset_make
from MarkDataset import Mark_Dataset
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train(epoch, model, train_loader, optimizer, mode='Faster_R-CNN'):
    print("=======Epoch:{}, mode:{}=======".format(epoch, mode))
    model.train()
    loss_classifier = 0.0
    loss_box_reg = 0.0
    loss_objectness = 0.0
    loss_rpn_box_reg = 0.0
    total_loss = 0.0
    for idx, images_targets in tqdm(enumerate(train_loader), total=len(train_loader)):
        images, targets = images_targets
        images = [image.cuda() for image in images]
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if mode == 'Faster_R-CNN':
            loss_classifier += loss_dict['loss_classifier'].item()
            loss_box_reg += loss_dict['loss_box_reg'].item()
            loss_objectness += loss_dict['loss_objectness'].item()
            loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()
        total_loss += losses.item()
    loss_classifier /= len(train_loader)
    loss_box_reg /= len(train_loader)
    loss_objectness /= len(train_loader)
    loss_rpn_box_reg /= len(train_loader)
    total_loss /= len(train_loader)

    train_result = OrderedDict({'Epoch': epoch,
                                'loss_objectness':  loss_objectness,
                                'loss_rpn_box_reg':  loss_rpn_box_reg,
                                'loss_classifier': loss_classifier,
                                'loss_box_reg': loss_box_reg,
                                'total_loss': total_loss})
    print('Epoch:', epoch, '\n',
          'loss_objectness:', loss_objectness, '\n',
          'loss_rpn_box_reg:', loss_rpn_box_reg, '\n',
          'loss_classifier:', loss_classifier, '\n',
          'loss_box_reg:', loss_box_reg, '\n',
          'total_loss:', total_loss)

    print('train epoch done!')
    return train_result


# four-fold cross validation

anchor_sizes = ((16,), (32,), (48,), (68,), (168,))
aspect_ratios = ((0.33, 0.5, 1.0, 2.0, 3.0),) * len(anchor_sizes)
anchor_generator = AnchorGenerator(
    anchor_sizes, aspect_ratios)


box_roi_pool = MultiScaleRoIAlign(
    featmap_names=['0', '1', '2', '3'],
    output_size=7,
    sampling_ratio=2)

patients_list = ['process_1', 'process_2', ..., 'process_16']  # patient file name list
for i in range(4):
    val_case_name = patients_list[4*i: 4*(i+1)]
    suffix = val_case_name[0][7:] + val_case_name[1][7:] + val_case_name[2][7:] + val_case_name[3][7:]  # name the model of each fold
    dataset_make(val_case_name)
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=2, min_size=550,
                                                     pretrained_backbone=True,
                                                     rpn_fg_iou_thresh=0.7, rpn_nms_thresh=0.6,
                                                     box_nms_thresh=0.6, image_mean=(0.4324, 0.4324, 0.4324),
                                                     image_std=(0.2850, 0.2850, 0.2850),
                                                     rpn_anchor_generator=anchor_generator, box_roi_pool=box_roi_pool)
    model = model.to(device)
    train_data_root = r'/data/ljm/LRCN/FRCNN_train'  # including JPEGImages and Annotations create by dataset make
    val_data_root = r'/data/ljm/LRCN/FRCNN_val'
    train_dataset = Mark_Dataset(train_data_root, 'train')
    val_dataset = Mark_Dataset(val_data_root, 'val')

    data_train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0003, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=0, last_epoch=-1)
    num_epochs = 120
    model_dict_store_path = '/data/ljm/LRCN/models'
    Logger_train = logger.Logger(model_dict_store_path, 'Faster_RCNN_train' + suffix)

    print('start for' + suffix)
    for epoch in range(1, num_epochs+1):
        print('train epoch start\n')
        train_result = train(epoch, model, data_train_loader, optimizer, mode='Faster_R-CNN')
        Logger_train.update(train_result)
        # update the learning rate
        lr_scheduler.step()
        print('train epoch done\n')
    torch.save(model, os.path.join(model_dict_store_path, 'fold' + suffix + '.pth'))
    print('model saved')
    print("That's it!" + suffix)
    print('==================================================')

print('all 4 fold done!')
