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


class Cifar10(data.Dataset):

    def __init__(self, split, args, logger, trans=None, c_aug=None, s_aug=None):
        assert split in ("train", "test", "val")
        self.args = args
        self.logger = logger
        self.split = split

        self.trans = trans
        self.c_aug = c_aug
        self.s_aug = s_aug

        self.logger.info(f"setup cifar {split} ==>")
        cifar_data = self.setup_cifar()
        self.dataset = cifar_data

    def setup_cifar(self):
        """
        args.cifar_root_dir
        """
        # load meta data
        meta_data_path = os.path.join(
            self.args.DATA.cifar_root_dir, self.args.DATA.cifar_meta_file)
        with open(meta_data_path, "rb") as pkl:
            meta_data = pickle.load(pkl)

        if self.split == "train":
            path_list = pathlib.Path(self.args.DATA.cifar_root_dir).glob(
                self.args.DATA.cifar_train_reg_exp)
            pickup_class = self.args.DATA.cifar_train_class
        elif self.split == "val":
            path_list = pathlib.Path(self.args.DATA.cifar_root_dir).glob(
                self.args.DATA.cifar_val_reg_exp)
            pickup_class = self.args.DATA.cifar_val_class
        else:
            path_list = pathlib.Path(self.args.DATA.cifar_root_dir).glob(
                self.args.DATA.cifar_test_reg_exp)
            pickup_class = self.args.DATA.cifar_test_class

        all_data = []
        all_labels = []
        for path in path_list:
            with open(str(path), "rb") as pkl:
                this_batch = pickle.load(pkl, encoding="bytes")
            """
            loaded data is (batch, 3*32*32), 3=(RGB)
            """
            use_keys = list(this_batch.keys())
            for key in use_keys:
                if isinstance(key, str):
                    continue
                this_batch[key.decode("ascii")] = this_batch[key]
                this_batch.pop(key)

            images = this_batch["data"].reshape(-1, 3, 32, 32)
            use_mask = np.isin(this_batch["labels"], pickup_class)

            if np.any(use_mask):
                use_images = images[use_mask]
                all_data.append(use_images)
                all_labels.append(this_batch["labels"][use_mask])

        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        self.logger.info("pick up {} class = {}, data num = {}".format(self.split, pickup_class, len(all_data)))

        cifar_data = {
            "data": all_data,
            "labels": all_labels,
            "meta_data": meta_data
        }

        return cifar_data

    def __len__(self):
        return len(self.dataset["data"])

    def pickup(self, index):
        data = self.dataset["data"][index]
        label = self.dataset["labels"][index]
        # transpose from (C, W, H) -> (W, H, C)
        data = data.transpose(1, 2, 0)

        if self.args.TRAIN.seff.supervise_method == "rotate":
            rotate_num = np.random.randint(4)
            for i in range(rotate_num):
                data = np.rot90(data)
            pseudo_label = rotate_num
        else:
            raise NotImplementedError

        if self.trans is not None:
            data = self.trans(data)


        # print(self.dataset["meta_data"], label)
        data = {"data": data, "real_label": label, "pseudo_label": pseudo_label,
                "label_name": self.dataset["meta_data"]["label_names"][label]}

        return data

    def __getitem__(self, index):
        picked_data = self.pickup(index)

        return picked_data
