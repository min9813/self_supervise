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
import torchvision
import lib.utils.average_meter as average_meter
import lib.utils.simclr_utils as simclr_utils
from tqdm import tqdm
from sklearn import linear_model
try:
    from apex import amp
except ImportError:
    pass


def make_directory(path):
    if os.path.exists(str(path)) is False:
        os.makedirs(str(path))


def train_epoch(wrappered_model, train_loader, optimizer, epoch, args, logger=None):
    wrappered_model.train()
    meter = average_meter.AverageMeter()
    train_loader.dataset.set_train()

    iter_num = len(train_loader)
    iter_since = time.time()
    since = iter_since

    # all_logits = []
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()

        meter.add_value("time_data", time.time() - since)
        input_x = data["data"]
        if args.TRAIN.self_supervised_method == "simclr" and not args.TRAIN.finetune_linear:
            input_x, pseudo_label = simclr_utils.setup_simclr(input_x, data["data2"])

        elif args.TRAIN.self_supervised_method == "rotate" or args.TRAIN.finetune_linear:
            pseudo_label = data["label"]
        since = time.time()
        loss, output = wrappered_model(input_x, pseudo_label)
        meter.add_value("time_f", time.time()-since)

        with torch.no_grad():
            _, pred = torch.max(output, dim=1)
            correct = np.mean(pred.cpu().numpy() == pseudo_label.numpy())

        if args.TRAIN.finetune_linear:
            meter.add_value("acc", correct)
        else:
            meter.add_value("pseudo_acc", correct)
        # all_logits.append(raw_logits.detach().cpu().numpy())

        since = time.time()
        if args.TRAIN.fp16:
            with amp.scale_loss(loss, [optimizer]) as scaled_loss:
                scaled_loss.backward()
        # loss.backward()
        else:
            loss.backward()
        optimizer.step()

        meter.add_value("time_b", time.time()-since)

        meter.add_value("loss_total", loss)

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
            if batch_idx >= 10:
                break
        since = time.time()
    # all_logits = np.concatenate(all_logits)
    train_info = meter.get_summary()

    return train_info


def valid_epoch(wrappered_model, train_loader, epoch, args, logger=None):
    wrappered_model.eval()

    meter = average_meter.AverageMeter()
    train_loader.dataset.set_train()
    iter_num = len(train_loader)
    iter_since = time.time()
    since = iter_since

    with torch.no_grad():
        save_num = 64
        now_num = 0
        infer_data = []
        # all_logits = []
        for batch_idx, data in tqdm(enumerate(train_loader), total=iter_num, desc="validation"):

            meter.add_value("time_data", time.time() - since)
            input_x = data["data"]
            if args.TRAIN.self_supervised_method == "simclr" and not args.TRAIN.finetune_linear:
                input_x, pseudo_label = simclr_utils.setup_simclr(input_x, data["data2"])

            elif args.TRAIN.self_supervised_method == "rotate" or args.TRAIN.finetune_linear:
                pseudo_label = data["label"]
            since = time.time()

            loss, output = wrappered_model(input_x, pseudo_label)
            meter.add_value("time_f", time.time()-since)
            _, pred = torch.max(output, dim=1)
            correct = np.mean(pred.cpu().numpy() == pseudo_label.numpy())
            if args.TRAIN.finetune_linear:
                meter.add_value("acc", correct)
            else:
                meter.add_value("pseudo_acc", correct)
            # all_logits.append(raw_logits.cpu().numpy())

            # since = time.time()

            # meter.add_value("time_b", time.time()-since)

            meter.add_value("loss_total", loss)

            # if now_num < save_num:
            # infer_data.append(input_x)
            now_num += save_num

            if args.debug:
                if batch_idx >= 10:
                    break
            since = time.time()
        # infer_data = torch.cat(infer_data, dim=0)[:save_num]
        # all_logits = np.concatenate(all_logits)
    train_info = meter.get_summary()
    # inference(wrappered_model.model, infer_data, args, epoch)

    return train_info


def valid_inference(net, train_loader, val_loader, epoch, args, logger):
    train_loader.dataset.set_eval()
    val_loader.dataset.set_eval()
    with torch.no_grad():
        logger.info("Inference data ... ")
        iter_num = len(train_loader)
        train_labels = []
        train_logits = []
        for idx, data in tqdm(enumerate(train_loader), total=iter_num, desc="train"):
            input_x = data["data"]
            label = data["real_label"]
            input_x = input_x.cuda()
            raw_logits = net(input_x)
            train_labels.append(label.numpy())
            train_logits.append(raw_logits.cpu().numpy())
            if args.debug:
                if idx >= 3:
                    break
        
        train_labels = np.concatenate(train_labels)
        train_logits = np.concatenate(train_logits)

        output_trn_feats = {
            "label": train_labels,
            "feature": train_logits
        }

        iter_num = len(val_loader)
        valid_labels = []
        valid_logits = []
        for idx, data in tqdm(enumerate(val_loader), total=iter_num, desc="valid"):
            label = data["real_label"]
            input_x = data["data"]
            input_x = input_x.cuda()
            raw_logits = net(input_x)
            valid_labels.append(label.numpy())
            valid_logits.append(raw_logits.cpu().numpy())
            if args.debug:
                if idx >= 3:
                    break
        valid_labels = np.concatenate(valid_labels)
        valid_logits = np.concatenate(valid_logits)

        output_val_feats = {
            "label": valid_labels,
            "feature": valid_logits
        }

    save_feature_dir = os.path.join(args.DATA.feature_root_dir, "epoch_{:0>5}".format(epoch))
    make_directory(save_feature_dir)
    save_feature_path = os.path.join(save_feature_dir, "train.pickle")
    with open(save_feature_path, "wb") as pkl:
        pickle.dump(output_trn_feats, pkl)
    save_feature_path = os.path.join(save_feature_dir, "valid.pickle")
    with open(save_feature_path, "wb") as pkl:
        pickle.dump(output_val_feats, pkl)
    # result = linear_eval(train_logits, train_labels,
    #                      valid_logits, valid_labels, args, logger)


def linear_eval(x_data, y_data, x_test, y_test, args, logger):
    estimater = linear_model.LogisticRegression(solver="sag", max_iter=1000)
    logger.info("fitting ...")
    estimater.fit(x_data, y_data)
    logger.info("finish, start inference ...")
    y_pred = estimater.predict(x_test)
    acc = np.mean(y_pred == y_test)
    logger.info("finish, acc = {}".format(acc))

    result = {"linear_acc": acc}

    return result


def inference(model, data, args, epoch):
    # data = (B, C, W, H)
    B, C, W, H = data.size()
    with torch.no_grad():
        data = data.to(args.device)
        output, mean, val = model(data)
        output = output.cpu()
        nrow = int(np.floor(np.sqrt(B)))
        show_output = output[:nrow*nrow]
    make_directory(args.image_save_dir)
    save_path = os.path.join(
        args.image_save_dir, "epoch_{:0>5}.jpg".format(epoch))
    torchvision.utils.save_image(
        show_output, save_path, nrow, normalize=True, range=(-1., 1.))
