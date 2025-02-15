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
import math
import subprocess
import logging
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import lib.embeddings.locally_linear_embedding as locally_linear_embedding
import lib.utils.average_meter as average_meter
import lib.lossfunction.metric as metric
import lib.embeddings as embeddings
from tqdm import tqdm
from . import knn


def check_equal(a, b):
    assert a == b, (a, b)


def location():
    frame = inspect.currentframe().f_back
    return os.path.basename(frame.f_code.co_filename), frame.f_code.co_name, frame.f_lineno


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
    print("sample_class_num:", sample_class_num, location())

    if hasattr(args.MODEL, "trn_embedder"):
        trn_embedder = args.MODEL.trn_embedder
        val_embedder = args.MODEL.trn_embedder
        # val_embedder.nca.A.data = val_embedder.nca.A.data.to("cpu")
        val_embedder.nca = val_embedder.nca.to("cpu")

    else:
        trn_embedder, val_embedder = embeddings.meta.get_embedder(args)

    print("embedder:", val_embedder, location())

    def calc_accuracy_stats(query_feats_all, query_labels_all, support_mean_feats_all, support_one_feats, support_all_feats, name=""):
        for method in ("cossim", "l2euc", "innerprod"):
            result_n = calc_accuracy(query_feats_all, query_labels_all,
                                     support_mean_feats_all, sampled_class, args, args.TEST.n_support, distance_metric=method)
            for key, value in result_n.items():
                meter.add_value("{}_{}{}".format(key, method, name), value)
            result_one = calc_accuracy(
                query_feats_all, query_labels_all, support_one_feats, sampled_class, args, num_s=1, distance_metric=method)
            for key, value in result_one.items():
                meter.add_value("{}_{}{}".format(key, method, name), value)

            result_knn = calc_accuracy(
                query_feats_all, query_labels_all, support_all_feats, sampled_class, args, num_s=args.TEST.n_support, distance_metric=method, is_knn=True)
            for key, value in result_knn.items():
                meter.add_value("{}_{}{}".format(key, method, name), value)


    for each_test_idx in tqdm(range(args.TEST.few_shot_n_test)):
        sampled_class = np.random.choice(
            sample_class_cands, sample_class_num, replace=False)
        assert len(sampled_class) == args.TEST.n_way
        query_feats_all = []
        query_labels_all = []
        support_mean_feats_all = []
        support_one_feats = []
        support_all_feats = []
        # print("sampled class:", sampled_class)
        for label in sampled_class:
            class_sample_num = len(class2feature[label])
            sampled_index = np.random.choice(np.arange(
                class_sample_num), sample_num, replace=False)
            assert len(sampled_index) == sample_num
            sampled_feats = class2feature[label][sampled_index]

            support_feats = sampled_feats[:args.TEST.n_support]
            query_feats = sampled_feats[args.TEST.n_support:]

            support_one_feats.append(support_feats[0])

            support_mean_feats = np.mean(support_feats, axis=0)
            query_feats_all.append(query_feats)
            query_labels_all.extend([label]*args.TEST.n_query)
            support_mean_feats_all.append(support_mean_feats)
            # support_feats = np.concatenate((support_feats, support_mean_feats[None]))
            support_all_feats.append(
                support_feats
            )

        query_feats_all = np.concatenate(query_feats_all)
        query_labels_all = np.array(query_labels_all)
        support_mean_feats_all = np.array(support_mean_feats_all)
        support_one_feats = np.array(support_one_feats)
        support_all_feats = np.array(support_all_feats)

        if args.MODEL.embedding_flag:
            if "lda" in str(val_embedder) or "nca" in str(val_embedder).lower():
                if "lda" in str(val_embedder):
                    features = embedding_one_episode_lda(
                        support_feats=support_all_feats,
                        query_feats=query_feats_all,
                        embedder=val_embedder,
                        method=args.MODEL.embedding_method,
                        is_svd=args.MODEL.is_lda_svd,
                        svd_dim=args.MODEL.lda_svd_dim
                    )
                    embed_name = "lda"

                else:
                    if args.MODEL.embedding_finetuning:
                        args.MODEL.is_instanciate_each_iter = True
                        trn_embedder_new, val_embedder_new = embeddings.meta.get_embedder(args)
                        args.MODEL.is_instanciate_each_iter = False

                        val_embedder_new.register_nca_state_dict(
                            base_nca_state_dict=trn_embedder.nca.state_dict()
                        )

                        val_embedder_here = val_embedder_new

                    else:
                        val_embedder_here = val_embedder

                    features = embedding_one_episode_nca(
                        support_feats=support_all_feats, 
                        query_feats=query_feats_all, 
                        embedder=val_embedder_here, 
                        args=args)

                    embed_name = "nca"

                for n_dim, each_dim_features in features.items():
                    support_embeddings = each_dim_features["support"]
                    query_embeddings = each_dim_features["query"]

                    support_mean_feats_embed_all = np.mean(support_embeddings, axis=1)
                    support_one_feats_embed = support_embeddings[:, 0]

                    calc_accuracy_stats(
                        query_feats_all=query_embeddings,
                        query_labels_all=query_labels_all,
                        support_mean_feats_all=support_mean_feats_embed_all,
                        support_one_feats=support_one_feats_embed,
                        support_all_feats=support_embeddings,
                        name="_{}_{}".format(embed_name, n_dim)
                    )

                calc_accuracy_stats(
                    query_feats_all=query_feats_all,
                    query_labels_all=query_labels_all,
                    support_mean_feats_all=support_mean_feats_all,
                    support_one_feats=support_one_feats,
                    support_all_feats=support_all_feats
                )

            else:
                support_all_feats, query_feats_all = embedding_one_episode(
                    support_feats=support_all_feats,
                    query_feats=query_feats_all,
                    embedder=val_embedder,
                    method=args.MODEL.embedding_method
                )

                support_one_feats = support_all_feats[:, 0]
                support_mean_feats_all = np.mean(support_all_feats, axis=1)

                calc_accuracy_stats(
                    query_feats_all=query_feats_all,
                    query_labels_all=query_labels_all,
                    support_mean_feats_all=support_mean_feats_all,
                    support_one_feats=support_one_feats,
                    support_all_feats=support_all_feats
                )

        else:
            calc_accuracy_stats(
                query_feats_all=query_feats_all,
                query_labels_all=query_labels_all,
                support_mean_feats_all=support_mean_feats_all,
                support_one_feats=support_one_feats,
                support_all_feats=support_all_feats
            )

    val_info = meter.get_summary()

    if val_embedder is not None and hasattr(val_embedder, "nca"):
        if val_embedder.nca.A.data.device != args.device:
            val_embedder.nca = val_embedder.nca.to(args.device)

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

        for method in ("cossim", "l2euc", "normal"):
            result_n = calc_accuracy((query_feats_all_m, query_feats_all_s), query_labels_all,
                                     (support_mean_feats_all_m,
                                      support_mean_feats_all_s),
                                     sampled_class, args, args.TEST.n_support)
            for key, value in result_n.items():
                meter.add_value("{}_{}".format(key, method), value)
        result_one = calc_accuracy(
            (query_feats_all_m, query_feats_all_s), query_labels_all,
            (support_one_feats_m, support_one_feats_s), sampled_class,
            args, num_s=1
        )
        for key, value in result_one.items():
            meter.add_value(key, value)

    val_info = meter.get_summary()

    return val_info


def embedding_one_episode(support_feats, query_feats, embedder, method="naive"):
    """
    support_feats: (n_class, n_support, dim)
    query_feats: (n_class*n_query, dim)
    """
    n_support, n_class = support_feats.shape[:2]
    # n_query, _ = query_feats.shape[:2]
    # if "lda" in str(embedder):
    #     pass

    # else:
    support_feats = support_feats.reshape(n_class*n_support, -1)

    support_feats = torch.Tensor(support_feats)
    query_feats = torch.Tensor(query_feats)
    
    support_embedding, query_embedding = embedder(
        support_vector=support_feats,
        query_vector=query_feats,
        method=method
    )
    # print("finish")
    support_embedding = support_embedding.numpy()
    query_embedding = query_embedding.numpy()

    support_embedding = support_embedding.reshape(n_class, n_support, -1)
    # query_embedding = query_embedding.reshape(n_class, n_query, -1)

    return support_embedding, query_embedding


def embedding_one_episode_lda(support_feats, query_feats, embedder, method="naive", is_svd=False, svd_dim=64, lamb=0.001):
    """
    support_feats: (n_support, n_class, dim)
    query_feats: (n_class*n_query, dim)
    """
    n_class, n_support = support_feats.shape[:2]
    # n_query, _ = query_feats.shape[:2]
    # if "lda" in str(embedder):
    #     pass

    # else:
    # support_feats = support_feats.reshape(n_class*n_support, -1)

    support_feats = torch.Tensor(support_feats)
    query_feats = torch.Tensor(query_feats)
    
    if is_svd:
        lda_loss, logit, pick_v, support_feats, query_feats = embedder(
            support_vector=support_feats,
            query_vector=query_feats,
            method=method,
            is_svd=is_svd,
            svd_dim=svd_dim,
            lamb=lamb,
            get_feats=True
        )

    else:
        lda_loss, logit, pick_v = embedder(
            support_vector=support_feats,
            query_vector=query_feats,
            method=method,
            is_svd=is_svd,
            lamb=lamb,
            svd_dim=svd_dim
        )

    transformed_feats = {}
    support_feats = support_feats.reshape(n_class*n_support, -1)
    if pick_v.shape[1] > 5:
        pick_v_dim_list = [2, 3, 5, pick_v.shape[1]]
    else:
        pick_v_dim_list = [2, 3, pick_v.shape[1]]

    for dim in pick_v_dim_list:
        # print(support_feats.shape, pick_v.shape)
        transformed_train_feats = torch.mm(support_feats, pick_v[:, -dim:])
        transformed_test_feats = torch.mm(query_feats, pick_v[:, -dim:])

        transformed_train_feats = transformed_train_feats.reshape(n_class, n_support, -1)
        transformed_feats[dim] = {
            "support": transformed_train_feats.numpy(),
            "query": transformed_test_feats.numpy()
        }
    # print("finish")
    # support_embedding = support_embedding.numpy()
    # query_embedding = query_embedding.numpy()

    # query_embedding = query_embedding.reshape(n_class, n_query, -1)

    return transformed_feats


def embedding_one_episode_nca(support_feats, query_feats, embedder, args):
    """
    support_feats: (n_support, n_class, dim)
    query_feats: (n_class*n_query, dim)
    """
    n_class, n_support = support_feats.shape[:2]
    # n_query, _ = query_feats.shape[:2]
    # if "lda" in str(embedder):
    #     pass

    # else:
    # support_feats = support_feats.reshape(n_class*n_support, -1)

    support_feats = torch.Tensor(support_feats)
    query_feats = torch.Tensor(query_feats)
    
    transformed_feats = {}

    if args.MODEL.embedding_finetuning:
        max_iter_list = [5, 10, 20, 30, 50]

    else:
        max_iter_list = [5]
    for max_iter in max_iter_list:
        nca_loss, logit, transformed_support_feats, transformed_query_feats = embedder(
                    support_vector=support_feats,
                    query_vector=query_feats,
                    query_label=None,
                    init_method=args.MODEL.init_nca_method,
                    max_batch_size=math.ceil(n_class*n_support)+10,
                    distance_method=args.MODEL.mds_metric_type,
                    lr=args.MODEL.nca_lr,
                    max_iter=max_iter,
                    stop_diff=args.MODEL.nca_stop_diff,
                    scale=args.MODEL.nca_scale,
                    get_feats=True,
                    stop_criteria=args.MODEL.stop_criteria
                )

        # print(support_feats.shape)

        transformed_support_feats = transformed_support_feats.reshape(n_class, n_support, -1)
        transformed_feats["dim{}-iter{:0>2}".format(2, max_iter)] = {
            "support": transformed_support_feats.detach().numpy(),
            "query": transformed_query_feats.detach().numpy()
        }
    # print("finish")
    # support_embedding = support_embedding.numpy()
    # query_embedding = query_embedding.numpy()

    # query_embedding = query_embedding.reshape(n_class, n_query, -1)

    return transformed_feats


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
    dist = metric.calc_mahalanobis_numpy(
        query_feats_m, support_feats_m, support_feats_s)

    # log coef = (S_n, 1)
    log_coef = 0.5 * np.log(support_feats_s).sum(axis=1, keepdims=True)

    # N(m, s) = 1 / (2pi)^k sqrt(det(s)) exp(- (x-m)^T s^-1 (x-m))
    dist = - dist + log_coef.T

    return dist


def calc_accuracy(query_feats_all, query_labels_all, support_feats_all, support_class, args, num_s, distance_metric="cossim", is_knn=False):
    result = {}
    if is_knn and len(support_feats_all.shape) == 3:
        n_class, n_support, D = support_feats_all.shape
        assert D == query_feats_all.shape[-1], support_feats_all.shape
        support_feats_all = support_feats_all.reshape(-1, D)
        support_class_for_knn = np.repeat(support_class, n_support)
        check_equal(len(support_class_for_knn), len(support_feats_all))
        name = "knn"

    else:
        name = "m"

    if distance_metric == "cossim":
        if isinstance(query_feats_all, tuple):
            query_feats_m, query_feats_s = query_feats_all
            support_feats_m, support_feats_s = support_feats_all
            distance_mat = calc_cossim_dist(query_feats_m, support_feats_m)
        else:
            distance_mat = calc_cossim_dist(query_feats_all, support_feats_all)

    elif distance_metric == "l2euc":
        if isinstance(query_feats_all, tuple):
            query_feats_m, query_feats_s = query_feats_all
            support_feats_m, support_feats_s = support_feats_all
            distance_mat = calc_l2_dist(query_feats_m, support_feats_m)
        else:
            distance_mat = calc_l2_dist(query_feats_all, support_feats_all)

    elif distance_metric == "normal":
        assert len(query_feats_all) == 2
        distance_mat = calc_normal_prob(query_feats_all, support_feats_all)

    elif distance_metric == "innerprod":
        if isinstance(query_feats_all, tuple):
            query_feats_m, query_feats_s = query_feats_all
            support_feats_m, support_feats_s = support_feats_all
            distance_mat = calc_innerprod(query_feats_m, support_feats_m)
        else:
            distance_mat = calc_innerprod(query_feats_all, support_feats_all)

    if is_knn:
        for topk in args.TEST.knn_num_ks:
            pred_label_knn = knn.knn_hard_from_distmat(
                distance_mat=-distance_mat,
                labels=support_class_for_knn,
                topk=topk
            )
            mean_acc = np.mean(pred_label_knn == query_labels_all)
            result["mean_{}_acc_{}".format(name, topk)] = mean_acc
            # for label in support_class:
            #     this_label_mask = (query_labels_all == label)
            #     this_label_acc = np.mean(pred_label[this_label_mask] == label)
            #     result["each_{}_{}_mean_acc_{}".format(name, label, topk)] = this_label_acc
    else:
        # print(distance_mat.shape, support_feats_all.shape, query_feats_all.shape)
        pred_label_index = np.argmax(distance_mat, axis=1)
        pred_label = support_class[pred_label_index]
        mean_acc = np.mean(pred_label == query_labels_all)
        result["mean_{}_acc_{}".format(name, num_s)] = mean_acc
        for label in support_class:
            this_label_mask = (query_labels_all == label)
            this_label_acc = np.mean(pred_label[this_label_mask] == label)
            result["each_{}_{}_mean_acc_{}".format(
                name, label, num_s)] = this_label_acc

    return result


def calc_cossim_dist(feature1, feature2):
    # print("norm1 {}:".format(feature1.shape), np.sqrt(np.sum(feature1 * feature1, axis=1, keepdims=True)))
    # print("norm2 {}:".format(feature2.shape), np.sqrt(np.sum(feature2 * feature2, axis=1, keepdims=True)))
    feature1_norm = np.sqrt(np.sum(feature1 * feature1, axis=1, keepdims=True))
    mask = feature1_norm[:, 0] > 1e-4
    feature1[mask] = feature1[mask] / feature1_norm[mask]
        
    feature2_norm = np.sqrt(np.sum(feature2 * feature2, axis=1, keepdims=True))
    mask = feature2_norm[:, 0] > 1e-4
    feature2[mask] = feature2[mask] / feature2_norm[mask]
    
    distance_mat = np.dot(feature1, feature2.T)
    return distance_mat


def calc_innerprod(feature1, feature2):
    distance_mat = np.dot(feature1, feature2.T)
    return distance_mat


def calc_l2_dist(feature1, feature2):
    # (Qn, D), (Sn, D)
    XX = np.sum(feature1*feature1, axis=1, keepdims=True)
    XY = np.dot(feature1, feature2.T)
    YY = np.sum(feature2*feature2, axis=1, keepdims=True).T
    # print(XX.shape, XY.shape, YY.shape, feature1.shape)

    dist = XX - 2 * XY + YY
    dist = np.sqrt(np.abs(dist))

    return -dist


def fewshot_eval_meta(raw_logits, n_support, n_query, meta_mode="cossim"):
    raw_logits = raw_logits[:, :, -(n_support+n_query):]
    B, n_way, n_sample, D = raw_logits.shape
    support_feats = raw_logits[:, :, :n_support]

    support_feats = torch.mean(support_feats, dim=2)  # (B, n_class, feat_dim)

    query_feats = raw_logits[:, :, n_support:]
    query_feats = query_feats.reshape(
        B, n_way*n_query, -1)
    label_episode = torch.arange(n_way,
                                 dtype=torch.long, device=query_feats.device)
    label_episode = label_episode.view(-1,
                                       1).repeat(B, n_query).reshape(-1)
    # if not self.training:
    #     print(label_episode)
    # # print(label)
    # # print(label_episode)
    # # print(label)
    #     label = label.reshape(B,  self.n_way, n_sample)
    # # # print(label)
    #     query_label = label[:, :, n_support:]
    #     query_label = query_label.reshape(B, self.n_way*n_query, -1).reshape(-1)
    #     print(query_label)
    #     sdfa
    if meta_mode == "cossim":
        logit = compute_cosine_similarity(
            support_feats, query_feats,
            n_way=n_way
        )
    elif meta_mode == "euc":
        logit = metric.calc_l2_dist_torch(
            query_feats, support_feats, dim=2
        )
        # logit = logit.permute(0, 2, 1)
        logit = logit.reshape(-1, n_way)
    elif meta_mode == "innerprod":
        logit = compute_cosine_similarity(
            support_feats,
            query_feats,
            is_norm=False
        )
    else:
        raise NotImplementedError

    _, pred = torch.max(logit, dim=1)
    correct = np.mean(pred.cpu().numpy() == label_episode.numpy())
    # loss = self.criterion(logit, label_episode)
    # log_p_y = F.log_softmax(logit, dim=1).view(B*n_way, n_query, -1)
    # loss_2 = -log_p_y.gather(2, label_episode.reshape(B*n_way, n_query, 1)).mean()
    # print(label_episode)
    # print(label)
    # sdfa

    return correct


def compute_cosine_similarity(support_feats, query_feats, logit_scale=8, n_way=5, is_norm=True):
    if is_norm:
        query_feats = F.normalize(query_feats, dim=2)
        support_feats = F.normalize(support_feats, dim=2)
    # print(query_feats.size(), support_feats.size())
    cossim = torch.bmm(query_feats, support_feats.permute(0, 2, 1))
    # assert cossim.max() <= 1.001, cossim.max()
    cossim = cossim * logit_scale
    # order in one batch = (C1, C1, C1, ..., C2, C2, C2, ...)
    # cossim = cossim.permute(0, 2, 1)
    # print(cossim.shape, query_feats.shape)
    cossim = cossim.reshape(-1, n_way)

    return cossim
