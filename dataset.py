'''
copy form wondervictor :)
'''

import os
import numpy as np
from torch.utils import data
import xml.etree.ElementTree as eTree
import torchvision.transforms as transforms
from PIL import Image
import numpy.random as npr


# labels_path = "/home/jiapeifeng/Segmentation/labels_graphcut/"
class PASCALVOC(data.Dataset):
    def __init__(self, imageset, data_dir, test_mode=False, flip=True, devkit=None):
        self._imageset = imageset  # 'trainval', 'val'
        self._data_path = data_dir
        self.classes = ('aeroplane', 'bicycle', 'bird',
                        'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                        'person', 'pottedplant', 'sheep', 'sofa', 'train',
                        'tvmonitor')
        self.img_ext = '.jpg'
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None}
        self._class_to_index = dict(list(zip(self.classes, range(len(self.classes)))))
        self.image_index = self._load_imageset_index()
        self.num_classes = len(self.classes)
        self._devkit_path = devkit if devkit is not None else 'VOCdevkit'
        if imageset != 'test':
            self._labels = dict([(x, self._load_class_label_from_index(x)) for x in self.image_index])
        self.toTensor = transforms.ToTensor()
        self.normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        if flip:
            self.add_flipped()
        self.test_mode = test_mode

    @staticmethod
    def flip(img):
        return img.transpose(Image.FLIP_LEFT_RIGHT)

    def add_flipped(self):
        flipped = [x + '_flipped' for x in self.image_index]
        self.image_index += flipped

    def __len__(self):
        return len(self.image_index)

    def _load_imageset_index(self):
        """ Load Image Index from File
        """
        imageset_file = os.path.join(self._data_path, 'ImageSets', 'Main', self._imageset + '.txt')
        assert os.path.exists(imageset_file), 'Path does not exists: {}'.format(imageset_file)

        with open(imageset_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def image_path_from_index(self, index):
        image_path = os.path.join(self._data_path, 'JPEGImages', index + self.img_ext)
        return image_path

    def image_path_at(self, i):
        return self.image_path_from_index(self.image_index[i])

    def _load_class_label_from_index(self, index):
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = eTree.parse(filename)
        objs = tree.findall('object')
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            objs = non_diff_objs
        num_objs = len(objs)

        gt_classes = np.zeros(num_objs, dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            cls = self._class_to_index[obj.find('name').text.lower().strip()]
            gt_classes[ix] = cls

        real_label = np.zeros(self.num_classes).astype(np.float32)
        for label in gt_classes:
            real_label[label] = 1

        return {'labels': real_label}

    def __getitem__(self, idx):

        img_name = self.image_index[idx]
        flipped = False
        if img_name.find('_flipped') != -1:
            img_name = img_name.replace('_flipped', '')
            flipped = True
        img_path = self.image_path_from_index(img_name)
        img = Image.open(img_path)
        w, h = img.size
        max_size = max(h, w)
        img_size = self.scales[npr.randint(0, len(self.scales))]
        ratio = float(img_size) / float(max_size)
        w = int(w * ratio)
        h = int(h * ratio)
        img = img.resize((w, h))

        if flipped:
            img = self.flip(img)
        img = np.array(img)
        im_shape = img.shape[1::-1]
        img = self.toTensor(img)
        img = self.normalizer(img)

        if self._imageset == 'test':
            return img, np.array(im_shape, dtype=np.float32)
        else:
            label = self._labels[img_name]['labels']
            return img, label, np.array(im_shape, dtype=np.float32)


