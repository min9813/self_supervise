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


class FeatureDataset(data.Dataset):

    def __init__(self, split, args, logger):
        assert split in ("train", "test", "val")
        self.args = args
        self.logger = logger
        self.split = split

        self.logger.info(f"setup feature dataset {split} ==>")
        dataset = self.setup_data()
        self.dataset = dataset

    def setup_data(self):
        """
        args.cifar_root_dir
        """
        # load meta data

        if self.split == "train":
            path_list = pathlib.Path(self.args.DATA.feature_root_dir).glob(
                self.args.DATA.feature_train_reg_exp)
            pickup_class = self.args.DATA.feature_train_class
        elif self.split == "val":
            path_list = pathlib.Path(self.args.DATA.feature_root_dir).glob(
                self.args.DATA.feature_val_reg_exp)
            pickup_class = self.args.DATA.feature_val_class
        else:
            path_list = pathlib.Path(self.args.DATA.feature_root_dir).glob(
                self.args.DATA.feature_test_reg_exp)
            pickup_class = self.args.DATA.feature_test_class
        # print(self.args.DATA.feature_root_dir, self.args.DATA.feature_train_reg_exp)
        path_list = list(path_list)
        if len(path_list) == 0:
            raise ValueError(
                f"Path \'{self.args.DATA.feature_root_dir}\' and regularize expressoin \'{self.args.DATA.feature_val_reg_exp}\' not exist")
        all_data = []
        all_labels = []
        for path in path_list:
            with open(str(path), "rb") as pkl:
                this_batch = pickle.load(pkl)
            """
            loaded data is (batch, 3*32*32), 3=(RGB)
            """

            features = this_batch["feature"]
            use_mask = np.isin(this_batch["label"], pickup_class)
            # print(path, pickup_class)
            if np.any(use_mask):
                use_features = features[use_mask]
                all_data.append(use_features)
                labels = np.array(this_batch["label"])
                all_labels.append(labels[use_mask])

        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        self.logger.info("pick up {} class = {}, data num = {}".format(
            self.split, pickup_class, len(all_data)))

        picked_data = {
            "data": all_data,
            "labels": all_labels,
        }

        return picked_data

    def __len__(self):
        return len(self.dataset["data"])

    def set_train(self):
        pass

    def set_eval(self):
        pass
 
    def pickup(self, index):
        data = self.dataset["data"][index]
        label = self.dataset["labels"][index]

        # print(self.dataset["meta_data"], label)
        data = {"data": data, "label": label}

        return data

    def __getitem__(self, index):
        picked_data = self.pickup(index)

        return picked_data
