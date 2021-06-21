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
import torch
from torch.autograd import detect_anomaly
import torchvision
import lib.utils.average_meter as average_meter
import lib.evaluation.few_shot_eval as few_shot_eval
from tqdm import tqdm
from sklearn import linear_model
try:
    from apex import amp
except ImportError:
    pass


def make_directory(path):
    if os.path.exists(str(path)) is False:
        os.makedirs(str(path))


def iter_func(wrappered_model, data, args, meter, since, optimizer=None, get_features=False):
    is_train = optimizer is not None
    if is_train:
        optimizer.zero_grad()

    meter.add_value("time_data", time.time() - since)
    input_x = data["data"]
    label = data["label"]
    since = time.time()
    # with detect_anomaly():
    output = wrappered_model(input_x, label)

    # loss, output = output
    loss = output["loss_total"]
    # logit = output["logit"]
    if is_train:
        meter.add_value("time_f", time.time()-since)
        since = time.time()
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        # loss.backward()
        else:
            loss.backward()
        optimizer.step()

        meter.add_value("time_b", time.time()-since)

    if "label_episode" in output:
        pseudo_label = output["label_episode"].cpu()

    else:
        pseudo_label = label
        
    for key, value in output.items():
        if key.startswith("loss"):
            # print(key, value)
            meter.add_value(key, value)

        elif key.endswith("ok"):
            meter.add_value(key, value)

        elif key.startswith("lda") and "logit" not in key:
            meter.add_value(key, value)

        elif key.endswith("logit"):
            with torch.no_grad():
                _, pred = torch.max(
                    value,
                    dim=1
                )
                acc = np.mean((pred.cpu() == pseudo_label).numpy())
                key = key.replace("logit", "acc")
                meter.add_value(key, acc)

    if get_features:
        features = output["features"]
        # print(features.shape, label.shape)
        # sdfa
        label = label.reshape(-1)
        return features.cpu(), label


def train_epoch(wrappered_model, train_loader, optimizer, epoch, args, logger=None):
    wrappered_model.train()
    meter = average_meter.AverageMeter()

    iter_num = len(train_loader)
    iter_since = time.time()
    since = iter_since

    # all_logits = []
    for batch_idx, data in enumerate(train_loader):
        iter_func(wrappered_model, data, args,
                  meter, since, optimizer=optimizer)
        if args.LOG.train_print and (batch_idx+1) % args.LOG.train_print_iter == 0:
            # current training accuracy
            time_cur = (time.time() - iter_since)
            meter.add_value("time_iter", time_cur)
            iter_since = time.time()

            msg = f"Epoch [{epoch}] [{batch_idx+1}]/[{iter_num}]\t"
            summary = meter.get_summary()
            for name, value in summary.items():
                msg += " {}:{:.6f} ".format(name, value)
            logger.info(msg)

        if args.debug:
            if batch_idx >= 5:
                break
        since = time.time()

    # all_logits = np.concatenate(all_logits)
    train_info = meter.get_summary()

    return train_info


def valid_epoch(wrappered_model, train_loader, epoch, args, logger=None, get_features=False, get_features_only=False):
    wrappered_model.eval()

    meter = average_meter.AverageMeter()
    iter_num = len(train_loader)
    iter_since = time.time()
    since = iter_since

    # with torch.no_grad():
    all_logits = []
    all_labels = []
    if args.MODEL.is_instanciate_each_iter:
        for param in wrappered_model.model.parameters():
            param.requires_grad = False
        for param in wrappered_model.head.parameters():
            param.requires_grad = False

    else:
        for param in wrappered_model.parameters():
            param.requires_grad = False

    for batch_idx, data in tqdm(enumerate(train_loader), total=iter_num, desc="validation"):
        # if get_features:
        features, labels = iter_func(wrappered_model, data, args, meter, since, get_features=True)
        all_logits.append(features)
        all_labels.append(labels)

        # else:
            # iter_func(wrappered_model, data, args, meter, since, get_features=get_features)

        if args.debug:
            if batch_idx >= 5:
                break
        since = time.time()
    # infer_data = torch.cat(infer_data, dim=0)[:save_num]
    # all_logits = np.concatenate(all_logits)
    # if get_features:
    #     all_labels = np.concatenate(all_labels)
    #     all_logits = np.concatenate(all_logits)

    all_labels = torch.cat(all_labels)
    all_logits = torch.cat(all_logits)        
    if args.TRAIN.self_supervised_method.startswith("supervise") and not get_features_only:

        result = few_shot_eval.evaluate_fewshot(
            all_logits.numpy(), all_labels.numpy(), args)

    else:
        result = None

    train_info = meter.get_summary()

    if result is not None:
        train_info.update(result)
    # inference(wrappered_model.model, infer_data, args, epoch)

    if args.MODEL.is_instanciate_each_iter:
        for param in wrappered_model.model.parameters():
            param.requires_grad = True

        for param in wrappered_model.head.parameters():
            param.requires_grad = True

    else:
        for param in wrappered_model.parameters():
            param.requires_grad = True

    if get_features:
        return train_info, all_logits, all_labels

    else:
        return train_info
