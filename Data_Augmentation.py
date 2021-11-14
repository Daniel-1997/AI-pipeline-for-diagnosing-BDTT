import random
import numpy as np
import cv2
import math
import torchvision.transforms as transforms

class DataAugmentForObjectDetection:
    def __init__(self, rotation_rate=0.4, max_rotation_angle=10,
                 crop_rate=0.4, shift_rate=0.4, flip_rate=0.5):
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.flip_rate = flip_rate


    # rotate use
    def _rotate_img_bbox(self, img, bboxes, scale=1.):
        '''
        rotation for the image and the bounding box
        input:
            img: array,(h,w,c)
            bboxes: bounding boxes,one list, [x_min, y_min, x_max, y_max],
            make sure x_min, x_max, y_min, y_max are numbers
            angle: angle for rotation
            scale:default 1
        output:
            rot_img: img_array after rotation
            rot_bboxes: bounding box after rotation
        '''
        w = img.shape[1]
        h = img.shape[0]

        angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # affine transformation
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        # ---------------------- rectify bbox coordinates ----------------------
        rot_bboxes = list()
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))

            concat = np.vstack((point1, point2, point3, point4))


            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx + rw
            ry_max = ry + rh

            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])

        return rot_img, rot_bboxes

    # crop use
    def _crop_img_bboxes(self, img, bboxes):
        '''
        cropping for the image and the bounding box
        input:
            img: array,(h,w,c)
            bboxes: bounding boxes, one list, [x_min, y_min, x_max, y_max],
                    make sure x_min, x_max, y_min, y_max are numbers
        output:
            crop_img: img_array after cropping
            crop bboxes: bounding box after cropping
        '''

        w = img.shape[1]
        h = img.shape[0]
        x_min = w
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min
        d_to_right = w - x_max
        d_to_top = y_min
        d_to_bottom = h - y_max

        crop_x_min = int(x_min - random.uniform(d_to_left//2, d_to_left))
        crop_y_min = int(y_min - random.uniform(d_to_top//2, d_to_top))
        crop_x_max = int(x_max + random.uniform(d_to_right//2, d_to_right))
        crop_y_max = int(y_max + random.uniform(d_to_bottom//2, d_to_bottom))

        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        crop_bboxes = list()
        for bbox in bboxes:
            crop_bboxes.append([bbox[0] - crop_x_min, bbox[1] - crop_y_min, bbox[2] - crop_x_min, bbox[3] - crop_y_min])

        return crop_img, crop_bboxes

    # shift use
    def _shift_pic_bboxes(self, img, bboxes):
        '''

        input:
            img: array,(h,w,c)
            bboxes: bounding boxes, one list, [x_min, y_min, x_max, y_max],
                    make sure x_min, x_max, y_min, y_max are numbers
        output:
            shift_img: img_array after shift
            shift_bboxes: bounding box after shift
        '''

        w = img.shape[1]
        h = img.shape[0]
        x_min = w
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min
        d_to_right = w - x_max
        d_to_top = y_min
        d_to_bottom = h - y_max

        x = random.uniform(-(d_to_left - 1) / 8, (d_to_right - 1) / 8)
        y = random.uniform(-(d_to_top - 1) / 8, (d_to_bottom - 1) / 8)

        M = np.float32([[1, 0, x], [0, 1, y]])
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        shift_bboxes = list()
        for bbox in bboxes:
            shift_bboxes.append([int(bbox[0] + x), int(bbox[1] + y), int(bbox[2] + x), int(bbox[3] + y)])

        return shift_img, shift_bboxes

    # flip use
    def _flip_pic_bboxes(self, img, bboxes):
        '''
            input:
                img: array,(h,w,c)
                bboxes: bounding boxes, one list, [x_min, y_min, x_max, y_max],
                make sure x_min, x_max, y_min, y_max are numbers
            output:
                flip_img: img_array after flipping
                flip_bboxes: bounding box after flipping
        '''
        import copy
        flip_img = copy.deepcopy(img)
        a = random.random()
        if a < 0.5:
            horizon_vertical = True
        else:
            horizon_vertical = False
        h, w, _ = img.shape
        if horizon_vertical:
            if a < 0.25:
                flip_img = cv2.flip(flip_img, 1)  # 1 for horizontal
            else:
                flip_img = cv2.flip(flip_img, 0)  # 0 for vertical

        flip_bboxes = list()
        for box in bboxes:
            x_min = box[0]
            y_min = box[1]
            x_max = box[2]
            y_max = box[3]
            if a < 0.25:
                flip_bboxes.append([w - x_max, y_min, w - x_min, y_max])
            elif horizon_vertical:
                flip_bboxes.append([x_min, h - y_max, x_max, h - y_min])
            else:
                flip_bboxes.append([x_min, y_min, x_max, y_max])

        return flip_img, flip_bboxes

    def dataAugment(self, img, bboxes):
        '''
        data augmentation
        input:
            img: array,(h,w,c)
            bboxes: bounding boxes, one list, [x_min, y_min, x_max, y_max],
                make sure x_min, x_max, y_min, y_max are numbers
        output:
            img: img_array after transformation
            bboxes: bounding box after transformation
        '''

        if random.random() < self.shift_rate:  # shift
            # print('translation')
            # change_num += 1
            img, bboxes = self._shift_pic_bboxes(img, bboxes)

        if random.random() < self.crop_rate:  # crop
            # print('cropping')
            # change_num += 1
            img, bboxes = self._crop_img_bboxes(img, bboxes)

        if random.random() < self.rotation_rate:  # rotate
            # print('rotation')
            # change_num += 1
            # angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
            scale = random.uniform(0.9, 1.0)
            img, bboxes = self._rotate_img_bbox(img, bboxes, scale)

        if random.random() < self.flip_rate:  # flip
            # print('flipping')
            # change_num += 1
            img, bboxes = self._flip_pic_bboxes(img, bboxes)
        # print('\n')
        # print('------')
        return img, bboxes

    def __call__(self, img, bboxes):
        img, bboxes = self.dataAugment(img, bboxes)
        _to_tensor = transforms.ToTensor()
        img = _to_tensor(img)
        return img, bboxes
