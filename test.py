import torch
import os
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as transforms

# evaluation on the control group and the validation set
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def showbbox(model, img, img_name, imgs_save_dir=None):
    """
    :param model: trained Faster R-CNN model
    :param img: torch tensor with a value range between 0 and 1
    :param img_name: image name for saving the image
    """
    model.eval()
    threshold = 0.3

    with torch.no_grad():
        '''
        prediction has a format of：
        [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
        [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'),
        'labels': tensor([1, 1], device='cuda:0'),
        'scores': tensor([1.0000, 1.0000], device='cuda:0')}]
        '''
        prediction = model([img.to(device)])

    if prediction[0]['boxes'].size(0) == 0:
        return None
    image = Image.open(os.path.join(imgs_dir, img_name))
    image = image.convert(mode='RGB')
    crop = transforms.CenterCrop((275, 421))
    image = crop(image)
    image = np.array(image)
    pred_score = list(prediction[0]['scores'].detach().cpu().numpy())
    pred_score_list = [pred_score.index(x) for x in pred_score if x > threshold]
    if len(pred_score_list) == 0:
        return None

    score = sorted(pred_score)[-1]

    dts = []
    for idx in pred_score_list:
        box = prediction[0]['boxes'][idx]
        xmin = round(box[0].item())
        ymin = round(box[1].item())
        xmax = round(box[2].item())
        ymax = round(box[3].item())
        # print(xmin, ymin, xmax, ymax)
        dts.append([xmin, ymin, xmax, ymax])
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 255), thickness=2)

    if imgs_save_dir is not None:
        cv2.imwrite(os.path.join(imgs_save_dir, img_name), image)

    return score, dts, len(pred_score_list)


model_path = '/data/ljm/LRCN/LITS2017肝脏肿瘤分割挑战数据集/fold_1_2_3_4.pth'   # need change
model = torch.load(model_path)
model.to(device)

img_file_dir = '/data/ljm/LRCN/case_CT'
img_file_list = ['process_1', 'process_2', 'process_3', 'process_4']         # need change
stat = {'score': [], 'name': [], 'num_tp_ratio': []}
for img_file in img_file_list:
    imgs_dir = os.path.join(img_file_dir,  img_file)
    num_total += len(os.listdir(imgs_dir))
    i = 0
    max_score = 0
    num_box = 0
    for img_name in os.listdir(imgs_dir):
        img_path = os.path.join(imgs_dir, img_name)
        img = Image.open(img_path)
        img = img.convert(mode='RGB')
        transform = transforms.Compose([transforms.CenterCrop((275, 421)), transforms.ToTensor()])
        img = transform(img)
        result = showbbox(model, img, img_name)
        if result is not None:
            i += 1
            num_box += result[2]
            max_score = max(max_score, result[0])
    stat['name'].append(img_file)
    stat['score'].append(max_score)
    stat['num_tp_ratio'].append(i/len(os.listdir(imgs_dir)))
scores = stat['score']
print(np.array(scores).mean())
print(stat)
