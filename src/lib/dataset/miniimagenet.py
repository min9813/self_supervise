import os
import sys
import pathlib
import time
import collections
import itertools
import shutil
import pickle
import inspect
import json
import subprocess
import logging
import numpy as np
import torch.utils.data as data
import cv2
from tqdm import tqdm
from torchvision.transforms import transforms
from PIL import Image
from .dataset_utils import cv2pil

class MiniImageNet(data.Dataset):

    def __init__(self, split, args, logger, trans=None, c_aug=None, s_aug=None):
        assert split in ("train", "test", "val")
        self.args = args
        self.logger = logger
        self.split = split

        self.trans = trans
        self.c_aug = c_aug
        self.s_aug = s_aug
        if self.c_aug is not None:
            if isinstance(self.c_aug, transforms.Compose):
                aug_class = "transform"
            else:
                aug_class = "imgaug"
        else:
            aug_class = "transform"
        self.aug_class = aug_class
        self.logger.info("### USE COLOR AUGUMENTATION ###")
        self.logger.info(str(c_aug))

        self.logger.info("### USE SHAPE AUGUMENTATION ###")
        self.logger.info(str(s_aug))

        self.mode = "train"

        self.logger.info(f"setup cifar {split} ==>")
        data = self.setup_data()
        self.class_num = len(data["meta_data"])
        self.dataset = data

    def setup_data(self):
        """
        args.data_root_dir
        """
        # load meta data
        class_json_path = os.path.join(
            self.args.DATA.data_root_dir, self.args.DATA.class_json_files[self.split])
        with open(class_json_path, "r") as f:
            class_info = json.load(f)

        print(class_json_path)

        self.class_info = class_info

        class_list = sorted(class_info.keys())
        name2label = {}
        for class_name in class_list:
            if class_name in name2label:
                continue
            label = len(name2label)
            name2label[class_name] = label

        if self.split == "train":
            path_list = pathlib.Path(self.args.DATA.data_root_dir).glob(
                self.args.DATA.data_train_reg_exp)
        elif self.split == "val":
            path_list = pathlib.Path(self.args.DATA.data_root_dir).glob(
                self.args.DATA.data_val_reg_exp)
        else:
            path_list = pathlib.Path(self.args.DATA.data_root_dir).glob(
                self.args.DATA.data_test_reg_exp)
        
        all_data = []
        for path in tqdm(path_list, total=len(class_info)*600):
            image = cv2.imread(str(path))
            h, w, c = image.shape
            assert h == self.args.DATA.image_size
            image = cv2pil(image)
            class_name = path.parent.stem
            # print(path)
            assert class_name in class_info, class_name
            label = name2label[class_name]
            data = {
                "image": image,
                "label": label,
                "label_code": class_name
            }
            all_data.append(data)

        self.logger.info("pick up {} class num = {}, data num = {}".format(
            self.split, len(class_info), len(all_data)))

        data_data = {
            "data": all_data,
            "meta_data": class_info
        }

        return data_data

    def __len__(self):
        return len(self.dataset["data"])

    def augment_simclr(self, data):
        if self.c_aug is not None:
            if self.aug_class == "transform":
                # transforms.Compose
                if not isinstance(data, Image.Image):
                    data = Image.fromarray(data)
                if self.s_aug is not None:
                    data1 = self.s_aug(data)
                data1 = self.c_aug(data1)
            else:
                # imageaug
                data1 = self.c_aug(image=data)
                if self.s_aug is not None:
                    data1 = self.s_aug(image=data1)
        else:
            assert self.args.TRAIN.self_supervised_method.startswith("supervise")
            data1 = data
        data1 = self.trans(data1)

        return data1

    def set_eval(self):
        self.mode = "eval"

    def set_train(self):
        self.mode = "train"

    def pickup(self, index):
        data = self.dataset["data"][index]
        image = data["image"]
        label = data["label"]
        label_code = data["label_code"]
        label_name = self.class_info[label_code]

        if self.args.TRAIN.self_supervised_method == "rotate":
            rotate_num = np.random.randint(4)
            for i in range(rotate_num):
                image = np.rot90(image).copy()

            if self.trans is not None:
                image = self.trans(image)
            pseudo_label = rotate_num
            picked_data = {
                "data": image,
                "label": pseudo_label,
                "label_name": label_name
            }
        elif self.args.TRAIN.self_supervised_method == "simclr":
            if self.mode == "train":
                image1 = self.augment_simclr(image)
                image2 = self.augment_simclr(image)
            elif self.mode == "eval":
                image1 = self.trans(image)
                image2 = -1
            pseudo_label = -1
            picked_data = {
                "data": image1,
                "data2": image2,
                "label": pseudo_label,
            }
        elif self.args.TRAIN.self_supervised_method.startswith("supervise"):
            if self.mode == "train":
                image1 = self.augment_simclr(image)
            elif self.mode == "eval":
                image1 = self.trans(image)
            picked_data = {
                "data": image1,
                "label": label
            }
        else:
            raise NotImplementedError

        # print(self.dataset["meta_data"], label)
        other_data = {"real_label": label,
                      "label_name": label_code}
        picked_data.update(other_data)

        return picked_data

    def __getitem__(self, index):
        picked_data = self.pickup(index)

        return picked_data
