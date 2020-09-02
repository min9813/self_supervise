import os
import sys
import pathlib
import logging
import time
import collections
import itertools
import shutil
import pickle
import inspect
import json
import subprocess
import numpy as np
import pandas as pd
import cv2
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import scipy
import lib.evaluation as evaluation
import lib.dataset as dataset
import lib.lossfunction as lossfunction
import lib.utils.logger_config as logger_config
import lib.utils.average_meter as average_meter
import lib.utils.epoch_func as epoch_func
import lib.utils.get_aug_and_trans as get_aug_and_trans
import lib.network as network
from torch.optim import lr_scheduler
from lib.utils.configuration import cfg as args
from lib.utils.configuration import cfg_from_file, format_dict
try:
    from apex import amp
    fp16 = True
except ImportError:
    fp16 = False


def fix_seed(seed=0):
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def make_directory(path):
    if os.path.exists(str(path)) is False:
        os.makedirs(str(path))


def train():
    if len(sys.argv) == 2:
        cfg_file = sys.argv[1]
        cfg_from_file(cfg_file)
    else:
        cfg_file = "default"

    if len(args.gpus.split(',')) > 1 and args.use_multi_gpu:
        multi_gpus = True
    else:
        multi_gpus = False
    args.multi_gpus = multi_gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    if args.is_cpu:
        print("cpu!!")
        args.device = torch.device("cpu")
    else:
        if args.multi_gpus:
            args.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            args.device = torch.device(args.cuda_id)
            print("use cuda id:", args.device)

    fix_seed(args.seed)

    make_directory(args.LOG.save_dir)

    config_file = pathlib.Path(cfg_file)
    stem = config_file.stem
    args.exp_version = stem

    parent = config_file.parent.stem
    args.exp_type = parent

    args.MODEL.save_dir = f"{args.MODEL.save_dir}/{args.exp_type}/{args.exp_version}"
    if args.TRAIN.finetune_linear is False:
        args.DATA.feature_root_dir = f"{args.DATA.feature_root_dir}/{args.exp_type}/{args.exp_version}"

    msglogger = logger_config.config_pylogger(
        './config/logging.conf', args.exp_version, output_dir="{}/{}".format(args.LOG.save_dir, parent))
    trn_logger = logging.getLogger().getChild('train')
    val_logger = logging.getLogger().getChild('valid')

    msglogger.info("#"*30)
    msglogger.info("#"*5 + "\t" + "CONFIG FILE: " + str(config_file))

    msglogger.info("#"*30)

    if args.debug:
        args.TRAIN.total_epoch = 500
        args.LOG.train_print_iter = 1

    args.TRAIN.fp16 = args.TRAIN.fp16 and fp16

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    trans, c_aug, s_aug = get_aug_and_trans.get_aug_trans_torch_strong(
        args.TRAIN.color_aug, args.TRAIN.shape_aug, mean=mean, std=std, image_size=84)
    mean = np.array(mean)[:, None, None]
    std = np.array(std)[:, None, None]
    if args.TRAIN.finetune_linear:
        trn_dataset = dataset.feature_dataset.FeatureDataset(
            "train", args, msglogger)

    else:
        if args.DATA.dataset == "cifar10":
            trn_dataset = dataset.cifar.Cifar10(
                "train", args, msglogger, trans, c_aug=c_aug, s_aug=s_aug)
        elif args.DATA.dataset == "miniimagenet":
            trn_dataset = dataset.miniimagenet.MiniImageNet(
                "val", args, msglogger, trans, c_aug=c_aug, s_aug=s_aug)
            args.has_same = True


    train_loader = torch.utils.data.DataLoader(
        trn_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    save_dir = "./check/simclr"
    make_directory(save_dir)
    # train_loader.dataset.set_eval()
    images_all = []
    for idx, data in enumerate(train_loader):
        image = data["data"][0]
        image = image * std + mean
        image = image.numpy() * 255
        # print(np.min(image), np.max(image))
        # fkldjgsklj
        image = image.transpose(1, 2, 0)
        image = image.astype(np.uint8)
        # image2 = ["data2"][0]
        image2 = data["data2"][0]
        image2 = image2 * std + mean
        image2 = image2.numpy() * 255
        image2 = image2.transpose(1, 2, 0)
        image2 = image2.astype(np.uint8)
        print(image.shape, image2.shape)
        image = np.concatenate((image, image2), axis=1)
        images_all.append(image)
        if idx > 10:
            break
    image = np.concatenate(images_all, axis=0)
    path = os.path.join(save_dir, "{:0>5}.jpg".format(idx))
    cv2.imwrite(path, image)

def debug_fewshot_eval():
    if len(sys.argv) == 2:
        cfg_file = sys.argv[1]
        cfg_from_file(cfg_file)
    else:
        cfg_file = "default"

    # features = np.random.randn(300, 32)
    sample_n = 30
    n_class = 20
    features = [np.random.randn(32) for i in range(n_class)]
    all_feats = []
    for feats in features:
        for j in range(sample_n):
            if j%2 == 0:
                all_feats.append(feats)
            else:
                all_feats.append(np.random.randn(32))
    
    # features = [feats+np.random.random()*1e-3 for feats in features for j in range(sample_n)]
    features = np.array(all_feats)
    print("feature shape:", features.shape)
    labels = [np.full(sample_n, i) for i in range(n_class)]
    labels = np.concatenate(labels)

    result = evaluation.few_shot_eval.evaluate_fewshot(features, labels, args)
    print(result)


if __name__ == "__main__":
    train()
    # debug_fewshot_eval()