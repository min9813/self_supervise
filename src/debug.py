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
import lib.dataset as dataset
import lib.lossfunction as lossfunction
import lib.utils.logger_config as logger_config
import lib.utils.average_meter as average_meter
import lib.utils.epoch_func as epoch_func
import lib.utils.get_aug_and_trans as get_aug_and_trans
import lib.network as network
from torch.optim import lr_scheduler
from lib.utils.configuration import cfg as args
from lib.utils.configuration import cfg_from_file, format_dict, check_parameters
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

    check_parameters(args)

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
    # trans, c_aug, s_aug = get_aug_and_trans.get_aug_trans(
    #     args.TRAIN.color_aug, args.TRAIN.shape_aug, mean=mean, std=std)
    trans, c_aug, s_aug = get_aug_and_trans.get_aug_trans_torch_strong(
        args.TRAIN.color_aug, args.TRAIN.shape_aug, mean=mean, std=std, image_size=args.DATA.image_size)
    if args.TRAIN.finetune_linear:
        trn_dataset = dataset.feature_dataset.FeatureDataset(
            "train", args, msglogger)
        val_dataset = dataset.feature_dataset.FeatureDataset(
            "val", args, msglogger)

    else:
        if args.DATA.dataset == "cifar10":
            trn_dataset = dataset.cifar.Cifar10(
                "train", args, msglogger, trans, c_aug=c_aug, s_aug=s_aug)
            val_dataset = dataset.cifar.Cifar10(
                "val", args, msglogger, trans, c_aug=c_aug, s_aug=s_aug)
        elif args.DATA.dataset == "miniimagenet":
            if args.DATA.is_episode:
                trn_dataset = dataset.miniimagenet_episode.MiniImageNet(
                    "train", args, msglogger, trans, c_aug=c_aug, s_aug=s_aug)
            else:
                trn_dataset = dataset.miniimagenet.MiniImageNet(
                    "train", args, msglogger, trans, c_aug=c_aug, s_aug=s_aug)
            val_dataset = dataset.miniimagenet.MiniImageNet(
                "val", args, msglogger, trans, c_aug=c_aug, s_aug=s_aug)
            args.has_same = True

    train_loader = torch.utils.data.DataLoader(
        trn_dataset, batch_size=args.DATA.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    # for data in train_loader:
    #     print(data["data"].size())
    # dsklfal
    # memory_loader = torch.utils.data.DataLoader(
    # trn_dataset, batch_size=args.DATA.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    #     print(data["data"].size())
    # fkldsj
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.DATA.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)

    size = 0.5
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    save_dir = "./check/simclr"
    make_directory(save_dir)
    # train_loader.dataset.set_eval()
    if args.DATA.is_episode:
        std = torch.Tensor(std)[None, :, None, None]
        mean = torch.Tensor(mean)[None, :, None, None]
    else:
        std = torch.Tensor(std)[:, None, None]
        mean = torch.Tensor(mean)[:, None, None]
    images_all = []
    for idx, data in enumerate(train_loader):
        if args.DATA.is_episode:
            tensors = data["data"]
            data_in_batch_idx = 0
            batch_label_names = data["label_name"]
            batch_label_names = list(zip(*batch_label_names))
            for each_data in tensors:
                labels = data["label"][data_in_batch_idx]
                label_name = batch_label_names[data_in_batch_idx]
                each_data = each_data * std + mean
                each_data = each_data.numpy() * 255
                each_data = each_data.astype(np.uint8)
                B, C, W, H = each_data.shape
                tensor = each_data.reshape(
                    args.DATA.n_class_train, args.DATA.nb_sample_per_class, C, H, W)
                tensor = tensor.transpose(0, 1, 3, 4, 2)
                tensor = np.ascontiguousarray(tensor)
                for class_index in range(tensor.shape[0]):
                    class_tensor = tensor[class_index]
                    print(label_name[class_index * tensor.shape[1]])
                    for each_index_per_class in range(args.DATA.nb_sample_per_class):
                        image = class_tensor[each_index_per_class]
                        data_idx = class_index * \
                            tensor.shape[1]+each_index_per_class
                        label = labels[data_idx]
                        image = cv2.putText(image, "{}:{}".format(label, label_name[data_idx]), (20, 20),
                                            font, size, (255, 255, 255), thickness=thickness)
                        class_tensor[each_index_per_class] = image
                    tensor[class_index] = class_tensor
                print(tensor.shape)
                tensor = np.concatenate(tensor, axis=1)
                print(tensor.shape)
                tensor = np.concatenate(tensor, axis=1)
                print(tensor.shape, "ok")
                images_all.append(tensor)
                data_in_batch_idx += 1
            break
        else:
            image = data["data"]

            image = image * std + mean
            image = image.numpy() * 255
            # print(np.min(image), np.max(image))
            # fkldjgsklj
            image = image.transpose(1, 2, 0)
            image = np.ascontiguousarray(image)
            image = image.astype(np.uint8)
            real_label = data["label_name"][0]
            label = data["real_label"][0].numpy()
            print(label, real_label)
            image = cv2.putText(image, "{}:{}".format(label, real_label), (20, 20),
                                font, size, (255, 255, 255), thickness=thickness)
            # image2 = ["data2"][0]
            if "data2" in data:
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
    print(path)
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
            if j % 2 == 0:
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
