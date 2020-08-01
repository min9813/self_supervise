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
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import lib.utils.average_meter as average_meter
from tqdm import tqdm


def evaluate_fewshot(features, labels, args):
    meter = average_meter.AverageMeter()
    class2feature = {}
    print("feature:", features.shape, "labels:", labels.shape)
    for idx, label in enumerate(labels):
        try:
            class2feature[label.item()].append(features[idx])
        except KeyError:
            class2feature[label.item()] = [features[idx]]

    class_list = np.array(list(class2feature.keys()))
    sample_num = args.TEST.n_support + args.TEST.n_query

    sample_class_cands = []
    for c in class2feature:
        class2feature[c] = np.array(class2feature[c])
        if len(class2feature[c]) > sample_num:
            sample_class_cands.append(c)
    sample_class_num = min(args.TEST.n_way, len(sample_class_cands))
    print("sample_class_num:", sample_class_num)

    for each_test_idx in tqdm(range(args.TEST.few_shot_n_test)):
        sampled_class = np.random.choice(
            sample_class_cands, sample_class_num, replace=False)
        assert len(sampled_class) == args.TEST.n_way
        query_feats_all = []
        query_labels_all = []
        support_mean_feats_all = []
        support_one_feats = []
        for label in sampled_class:
            class_sample_num = len(class2feature[label])
            sampled_index = np.random.choice(np.arange(
                class_sample_num), sample_num, replace=False)
            assert len(sampled_index) == sample_num
            sampled_feats = class2feature[label][sampled_index]

            support_feats = sampled_feats[:args.TEST.n_support]
            query_feats = sampled_feats[args.TEST.n_support:]
            support_one_feats.append(support_feats[0])

            support_feats = np.mean(support_feats, axis=0)
            query_feats_all.append(query_feats)
            query_labels_all.extend([label]*args.TEST.n_query)
            support_mean_feats_all.append(support_feats)

        query_feats_all = np.concatenate(query_feats_all)
        query_labels_all = np.array(query_labels_all)
        support_mean_feats_all = np.array(support_mean_feats_all)
        support_one_feats = np.array(support_one_feats)

        result_n = calc_accuracy(query_feats_all, query_labels_all,
                                 support_mean_feats_all, sampled_class, args, args.TEST.n_support)
        for key, value in result_n.items():
            meter.add_value(key, value)
        result_one = calc_accuracy(
            query_feats_all, query_labels_all, support_one_feats, sampled_class, args, num_s=1)
        for key, value in result_one.items():
            meter.add_value(key, value)

    val_info = meter.get_summary()

    return val_info


def calc_accuracy(query_feats_all, query_labels_all, support_feats, support_class, args, num_s):
    result = {}
    if args.TEST.distance_metric == "cossim":
        distance_mat = calc_cossim_dist(query_feats_all, support_feats)
    elif args.TEST.distance_metric == "l2euc":
        distance_mat = calc_l2_dist(query_feats_all, support_feats)
    pred_label_index = np.argmax(distance_mat, axis=1)
    pred_label = support_class[pred_label_index]
    mean_acc = np.mean(pred_label == query_labels_all)
    result["mean_acc_{}".format(num_s)] = mean_acc
    for label in support_class:
        this_label_mask = (query_labels_all == label)
        this_label_acc = np.mean(pred_label[this_label_mask] == label)
        result["each_{}_mean_acc_{}".format(label, num_s)] = this_label_acc

    return result


def calc_cossim_dist(feature1, feature2):
    feature1 = feature1 / \
        np.sqrt(np.sum(feature1 * feature1, axis=1, keepdims=True))
    feature2 = feature2 / \
        np.sqrt(np.sum(feature2 * feature2, axis=1, keepdims=True))
    distance_mat = np.dot(feature1, feature2.T)
    return distance_mat


def calc_l2_dist(feature1, feature2):
    # (Qn, D), (Sn, D)
    XX = np.sum(feature1*feature1, axis=1, keepdims=True)
    XY = np.dot(feature1, feature2.T)
    YY = np.sum(feature2*feature2, axis=1, keepdims=True).T
    # print(XX.shape, XY.shape, YY.shape, feature1.shape)

    dist = XX - 2 * XY + YY
    dist = np.sqrt(dist)

    return -dist
