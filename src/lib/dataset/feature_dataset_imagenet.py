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

    def __init__(self, features, split, args, logger):
        assert split in ("train", "test", "val")
        self.args = args
        self.logger = logger
        self.split = split

        self.dataset = features
        # features = {"feature": np.ndarray, "label": np.ndarray}


    def __len__(self):
        return len(self.dataset["feature"])

    def set_train(self):
        pass

    def set_eval(self):
        pass
 
    def pickup(self, index):
        data = self.dataset["feature"][index]
        label = self.dataset["label"][index]

        # print(self.dataset["meta_data"], label)
        data = {"data": data, "label": label}

        return data

    def __getitem__(self, index):
        picked_data = self.pickup(index)

        return picked_data
