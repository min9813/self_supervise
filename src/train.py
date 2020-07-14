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
from lib.utils.configuration import cfg_from_file, format_dict
try:
    from apex import amp
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
    trans, c_aug, s_aug = get_aug_and_trans.get_aug_trans(
        args.TRAIN.color_aug, args.TRAIN.shape_aug, mean=mean, std=std)

    if args.TRAIN.finetune_linear:
        trn_dataset = dataset.feature_dataset.FeatureDataset(
            "train", args, msglogger)
        val_dataset = dataset.feature_dataset.FeatureDataset(
            "val", args, msglogger)

    else:
        trn_dataset = dataset.cifar.Cifar10("train", args, msglogger, trans, c_aug=c_aug, s_aug=s_aug)
        val_dataset = dataset.cifar.Cifar10("val", args, msglogger, trans, c_aug=c_aug, s_aug=s_aug)

    train_loader = torch.utils.data.DataLoader(
        trn_dataset, batch_size=args.DATA.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.DATA.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)

    # for data in val_loader:
    #     print(data["data"].size())
    # kjsglkfdj

    for key, value in vars(args).items():
        if isinstance(value, dict):
            for key2, value2 in value.items():
                msglogger.debug("{}:{}:{}".format(key, key2, value2))
        else:
            msglogger.debug("{}:{}".format(key, value))

    if args.TRAIN.finetune_linear:
        if len(args.MODEL.linear_layers):
            net = network.head.MLP(512, args.num_classes,
                                   args.MODEL.linear_layers)
        else:
            net = network.head.Head(512, args.num_classes)
        msg = "   HEAD   "
        msglogger.info("#"*15+msg + "#"*15)
        msglogger.info(str(net))
        msglogger.info("#"*(30+len(msg)))
    else:
        if args.MODEL.network == "resnet18":
            net = network.resnet.resnet18(pretrained=False)

        if len(args.MODEL.linear_layers):
            head = network.head.MLP(
                512, args.num_classes, args.MODEL.linear_layers)
        else:
            head = network.head.Head(512, args.num_classes)
        msg = "   HEAD   "
        msglogger.info("#"*15+msg + "#"*15)
        msglogger.info(str(head))
        msglogger.info("#"*(30+len(msg)))

    if args.MODEL.resume:
        net, start_epoch = network.model_io.load_model(
            net, args.MODEL.resume_net_path, logger=msglogger)
        args.TRAIN.start_epoch = start_epoch

    if args.TRAIN.self_supervised_method == "simclr" and not args.TRAIN.finetune_linear:
        criterion = lossfunction.simclr_loss.SimCLRLoss(
            args.DATA.batch_size, scale=args.TRAIN.logit_scale, device=args.device)
    else:
        criterion = nn.CrossEntropyLoss()

    if args.TRAIN.finetune_linear:
        wrapper = network.wrapper.LossWrapLinear(args, net, criterion)
    else:
        if args.TRAIN.self_supervised_method == "simclr":
            wrapper = network.wrapper.LossWrapSimCLR(args, net, head, criterion)
        else:
            wrapper = network.wrapper.LossWrap(args, net, head, criterion)

    wrapper = wrapper.cuda()

    if args.run_mode == "test":
        pass
    elif args.run_mode == "train":

        if args.OPTIM.optimizer == "adam":
            parameters = [
                {"params": net.parameters(), "lr": args.OPTIM.lr,
                 "weight_deacy": 5e-4},
            ]
            if not args.TRAIN.finetune_linear:
                parameters.append(
                    {"params": head.parameters(), "lr": args.OPTIM.lr,
                     "weight_deacy": 5e-4},
                )
            optimizer = torch.optim.Adam(
                parameters
            )
        elif args.TRAIN.optimizer == "sgd":
            parameters = [
                {"params": net.parameters(), "lr": args.OPTIM.lr,
                 "weight_deacy": 5e-4, "momentum": 0.9},
            ]
            if not args.TRAIN.finetune_linear:
                parameters.append(
                    {"params": head.parameters(), "lr": args.OPTIM.lr,
                     "weight_deacy": 5e-4, "momentum": 0.9},
                )
            optimizer = torch.optim.SGD(
                parameters
            )
        else:
            raise NotImplementedError

        if args.OPTIM.lr_scheduler == 'multi-step':
            milestones = args.OPTIM.lr_milestones
            scheduler = lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=args.OPTIM.lr_gamma, last_epoch=-1)
        elif args.OPTIM.lr_scheduler == 'cosine-anneal':
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.OPTIM.lr_tmax, eta_min=args.OPTIM.lr * 0.01, last_epoch=-1)
        elif args.OPTIM.lr_scheduler == 'patience':
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=args.OPTIM.lr_reduce_mode, factor=args.OPTIM.lr_gamma, patience=args.OPTIM.lr_patience,
                verbose=True, min_lr=args.OPTIM.lr_min, cooldown=args.OPTIM.lr_cooldown
            )
        elif args.OPTIM.lr_scheduler == "no":
            scheduler = None
        else:
            raise NotImplementedError

        if args.MODEL.resume:
            msglogger.info(f"Load optimizer from {args.MODEL.resume_opt_path}")
            checkpoint = torch.load(args.MODEL.resume_opt_path)
            optimizer.load_state_dict(checkpoint["optimizer"])

        args.lr = args.OPTIM.lr

        best_score = -1
        best_iter = -1
        for epoch in range(args.TRAIN.start_epoch, args.TRAIN.total_epoch+1):
            iter_since = time.time()
            msglogger.info("Start epoch {}".format(epoch))
            trn_info = epoch_func.train_epoch(
                wrapper, train_loader, optimizer, epoch, args, logger=trn_logger)
            val_info = epoch_func.valid_epoch(
                wrapper, val_loader, epoch, args, logger=val_logger)
            if not args.TRAIN.finetune_linear:
                if epoch % args.DATA.feature_save_freq == 0 and epoch > 0:
                    epoch_func.valid_inference(
                        wrapper.model, train_loader, val_loader, epoch, args, val_logger)
            # val_result = epoch_func.valid_inference(
            #     wrapper.model, train_loader, val_loader, args, val_logger)
            for param_group in optimizer.param_groups:
                lr = param_group["lr"]
                break

            # score = val_result["linear_acc"]
            if args.TRAIN.finetune_linear:
                score = val_info["acc"]
            else:
                score = val_info["pseudo_acc"]
            iter_end = time.time() - iter_since
            iter_since = time.time()
            msg = "Epoch:[{}/{}] lr:{} elapsed_time:{:.4f}s".format(epoch,
                                               args.TRAIN.total_epoch, lr, iter_end)
            # for name, value in val_result.items():
            #     msg += "{}:{:.4f} ".format(name, value)
            msglogger.info(msg)

            msg = "Valid: "
            for name, value in val_info.items():
                msg += "{}:{:.4f} ".format(name, value)
            msglogger.info(msg)

            msg = "TRAIN: "
            for name, value in trn_info.items():
                msg += "{}:{:.4f} ".format(name, value)
            msglogger.info(msg)

            is_best = best_score < score
            if is_best:
                best_score = score
                best_iter = epoch
            network.model_io.save_model(wrapper, optimizer, val_info, is_best, epoch,
                                        logger=msglogger, multi_gpus=args.multi_gpus,
                                        model_save_dir=args.MODEL.save_dir, delete_old=args.MODEL.delete_old)
            if scheduler is not None:
                if args.OPTIM.lr_scheduler == 'patience':
                    scheduler.step(score)
                elif args.OPTIM.lr_scheduler == "multi-step":
                    scheduler.step()
                else:
                    raise NotImplementedError
            """
            add  
            network.model_io.save_model(wrapper, optimizer, score, is_best, epoch, 
                                        logger=msglogger, multi_gpus=args.multi_gpus, 
                                        model_save_dir=args.model_save_dir, delete_old=args.delete_old)
            """
            if args.debug:
                if epoch >= 2:
                    break

        msglogger.info("Best Iter = {} loss={:.4f}".format(
            best_iter, best_score))


if __name__ == "__main__":
    train()
