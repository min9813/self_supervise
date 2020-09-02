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
import lib.lossfunction.metric as metric
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


def evaluate_fewshot_vae(z_mean, z_sigma, labels, args):
    meter = average_meter.AverageMeter()
    class2feature = {}
    # print("feature:", z_mean.shape, "labels:", labels.shape)
    for idx, label in enumerate(labels):
        label = label.item()
        try:
            class2feature[label]
        except KeyError:
            class2feature[label] = {}
        try:
            class2feature[label]["mean"].append(z_mean[idx])
            class2feature[label]["sigma"].append(z_sigma[idx])
        except KeyError:
            class2feature[label]["mean"] = [z_mean[idx]]
            class2feature[label]["sigma"] = [z_sigma[idx]]

    class_list = np.array(list(class2feature.keys()))
    sample_num = args.TEST.n_support + args.TEST.n_query

    sample_class_cands = []
    for c in class2feature:
        class2feature[c]["mean"] = np.array(class2feature[c]["mean"])
        class2feature[c]["sigma"] = np.array(class2feature[c]["sigma"])
        if len(class2feature[c]["mean"]) > sample_num:
            sample_class_cands.append(c)
    sample_class_num = min(args.TEST.n_way, len(sample_class_cands))
    # print("sample_class_num:", sample_class_num)

    for each_test_idx in tqdm(range(args.TEST.few_shot_n_test)):
        # print(sample_class_cands, sample_class_num)
        sampled_class = np.random.choice(
            sample_class_cands, sample_class_num, replace=False)
        assert len(sampled_class) == args.TEST.n_way
        query_feats_all_m = []
        query_feats_all_s = []
        query_labels_all = []
        support_mean_feats_all_m = []
        support_mean_feats_all_s = []
        support_one_feats_m = []
        support_one_feats_s = []
        for label in sampled_class:
            class_sample_num = len(class2feature[label]["mean"])
            sampled_index = np.random.choice(np.arange(
                class_sample_num), sample_num, replace=False)
            assert len(sampled_index) == sample_num
            sampled_feats_m = class2feature[label]["mean"][sampled_index]
            sampled_feats_s = class2feature[label]["sigma"][sampled_index]

            support_feats_m = sampled_feats_m[:args.TEST.n_support]
            support_feats_s = sampled_feats_s[:args.TEST.n_support]
            query_feats_m = sampled_feats_m[args.TEST.n_support:]
            query_feats_s = sampled_feats_s[args.TEST.n_support:]
            support_one_feats_m.append(support_feats_m[0])
            support_one_feats_s.append(support_feats_s[0])

            # support_feats = np.mean(support_feats_m, axis=0)
            support_feats_m, support_feats_s = aggregate_normal_stats(
                support_feats_m, support_feats_s, method=args.TEST.agg_stats_method)
            query_feats_all_m.append(query_feats_m)
            query_feats_all_s.append(query_feats_s)
            query_labels_all.extend([label]*args.TEST.n_query)
            support_mean_feats_all_m.append(support_feats_m)
            support_mean_feats_all_s.append(support_feats_s)

        query_feats_all_m = np.concatenate(query_feats_all_m)
        query_feats_all_s = np.concatenate(query_feats_all_s)
        query_labels_all = np.array(query_labels_all)
        support_mean_feats_all_m = np.array(support_mean_feats_all_m)
        support_mean_feats_all_s = np.array(support_mean_feats_all_s)
        support_one_feats_m = np.array(support_one_feats_m)
        support_one_feats_s = np.array(support_one_feats_s)

        result_n = calc_accuracy((query_feats_all_m, query_feats_all_s), query_labels_all,
                                 (support_mean_feats_all_m,
                                  support_mean_feats_all_s),
                                 sampled_class, args, args.TEST.n_support)
        for key, value in result_n.items():
            meter.add_value(key, value)
        result_one = calc_accuracy(
            (query_feats_all_m, query_feats_all_s), query_labels_all,
            (support_one_feats_m, support_one_feats_s), sampled_class,
            args, num_s=1
        )
        for key, value in result_one.items():
            meter.add_value(key, value)

    val_info = meter.get_summary()

    return val_info


def calc_minimum_weight(sigma_array):
    weight = 1 / (np.sum(1 / (sigma_array), axis=0) * sigma_array)
    return weight


def calc_stats_propose(x_mean, x_sigma):
    # mean = (N, D)
    weight = calc_minimum_weight(x_sigma)
    agg_mean = np.sum(x_mean * weight, axis=0)
    dist = (x_mean - agg_mean)
    agg_sigma = dist * dist + x_sigma
    agg_sigma = np.sum(weight * agg_sigma, axis=0)

    return agg_mean, agg_sigma


def calc_stats_vae_few(x_mean, x_sigma):
    weight = calc_minimum_weight(x_sigma)
    agg_mean = np.sum(x_mean * weight, axis=0, keepdims=True)
    agg_sigma = x_sigma.shape[0] / np.sum(1 / x_sigma)
    return agg_mean, agg_sigma


def aggregate_normal_stats(mean, sigma, method="propose"):
    if method == "propose":
        mean, sigma = calc_stats_propose(mean, sigma)
    elif method == "vae-few":
        mean, sigma = calc_stats_vae_few(mean, sigma)
    else:
        sdkfajl
    return mean, sigma


def calc_normal_prob(query_feats_all, support_feats_all):
    # query_feats = (Q_n, D), support_feats = (S_n, D)
    query_feats_m, query_feats_s = query_feats_all
    support_feats_m, support_feats_s = support_feats_all
    # coefed_sfeats_m = support_feats_m / support_feats_s

    # dist = (Q_n, S_n)
    # print(coefed_sfeats_m.shape, query_feats_m.shape, support_feats_m.shape, support_feats_s.shape)
    # dist = calc_l2_dist(query_feats_m, coefed_sfeats_m)
    dist = metric.calc_mahalanobis_numpy(query_feats_m, support_feats_m, support_feats_s)

    # log coef = (S_n, 1)
    log_coef = -0.5 * np.log(support_feats_s).sum(axis=1, keepdims=True)

    # N(m, s) = 1 / (2pi)^k sqrt(det(s)) exp(- (x-m)^T s^-1 (x-m))
    dist = dist + log_coef.T

    return dist


def calc_accuracy(query_feats_all, query_labels_all, support_feats, support_class, args, num_s):
    result = {}
    if args.TEST.distance_metric == "cossim":
        distance_mat = calc_cossim_dist(query_feats_all, support_feats)
    elif args.TEST.distance_metric == "l2euc":
        distance_mat = calc_l2_dist(query_feats_all, support_feats)
    elif args.TEST.distance_metric == "normal":
        assert len(query_feats_all) == 2
        distance_mat = calc_normal_prob(query_feats_all, support_feats)
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
