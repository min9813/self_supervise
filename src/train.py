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
import lib.utils.epoch_func as epoch_func_normal
import lib.utils.epoch_func_integrate as epoch_func_integrate
import lib.utils.get_aug_and_trans as get_aug_and_trans
import lib.utils.io_utils as io_utils
import lib.network as network
import lib.embeddings as embeddings
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
    args.fp16 = args.TRAIN.fp16

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

    elif args.MODEL.is_nca_train_two_stage and args.MODEL.embedding_flag and args.MODEL.embedding_algorithm == "nca":
        # train_loader = torch.utils.data.DataLoader(
            # trn_dataset, batch_size=args.DATA.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

        resume_net_path_stem = pathlib.Path(args.MODEL.resume_net_path).parent.stem
        load_train_feature_path = os.path.join(
            args.DATA.load_feature_dir,
            args.DATA.dataset,
            resume_net_path_stem+"_feature_train.pickle"
        )
        load_val_feature_path = os.path.join(
            args.DATA.load_feature_dir,
            args.DATA.dataset,
            resume_net_path_stem+"_feature_train.pickle"
        )

        if os.path.exists(load_train_feature_path):
            trn_feature_dict = io_utils.load_pickle(load_train_feature_path)
            val_feature_dict = io_utils.load_pickle(load_val_feature_path)

        else:
            raise FileNotFoundError("path {} not found!".format(load_train_feature_path))

        trn_dataset = dataset.feature_dataset_imagenet.FeatureDataset(
            features=trn_feature_dict,
            split="train",
            args=args,
            logger=msglogger
        )
        val_dataset = dataset.feature_dataset_imagenet.FeatureDataset(
            features=val_feature_dict,
            split="val",
            args=args,
            logger=msglogger
        )
        valid_batch_size = args.DATA.batch_size

    else:
        if args.DATA.dataset == "cifar10":
            trn_dataset = dataset.cifar.Cifar10(
                "train", args, msglogger, trans, c_aug=c_aug, s_aug=s_aug)
            val_dataset = dataset.cifar.Cifar10(
                "val", args, msglogger, trans, c_aug=c_aug, s_aug=s_aug)
            valid_batch_size = args.DATA.batch_size

        elif args.DATA.dataset == "miniimagenet":
            if args.DATA.is_episode:
                trn_dataset = dataset.miniimagenet_episode.MiniImageNet(
                    "train", args, msglogger, trans, c_aug=c_aug, s_aug=s_aug)
                # valid_batch_size = args.TRAIN.n_way * \
                # (args.TRAIN.n_support + args.TRAIN.n_query) * args.DATA.batch_size
                valid_batch_size = args.TRAIN.n_way // args.TEST.n_way
                val_dataset = dataset.miniimagenet_episode.MiniImageNet(
                    "val", args, msglogger, trans, c_aug=c_aug, s_aug=s_aug)
            else:
                trn_dataset = dataset.miniimagenet.MiniImageNet(
                    "train", args, msglogger, trans, c_aug=c_aug, s_aug=s_aug)
                valid_batch_size = args.DATA.batch_size
                val_dataset = dataset.miniimagenet.MiniImageNet(
                    "val", args, msglogger, trans, c_aug=c_aug, s_aug=s_aug)
            args.has_same = True


            # trn_result, trn_logits, trn_labels = epoch_func.valid_epoch(
            #     wrapper, train_loader, epoch, args, 
            #     logger=val_logger, 
            #     get_features=True, 
            #     get_features_only=True)
            # feature_dict = {
            #     "feature": trn_logits,
            #     "label": trn_labels
            # }
            # io_utils.save_pickle(
            #     path=load_feature_path,
            #     data=feature_dict
            # )

    # else:
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
        val_dataset, batch_size=valid_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)

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
            net = network.resnet.resnet18(pretrained=False, output_dim=args.MODEL.output_dim)
            # print(net)
            # sdfa
            feature_dim = args.MODEL.output_dim
        
        elif args.MODEL.network == "resnet12":
            net = network.resnet.ResNet12(output_dim=args.MODEL.output_dim)
            feature_dim = args.MODEL.output_dim

        elif args.MODEL.network == "resnet34":
            net = network.resnet.resnet34(pretrained=False, output_dim=args.MODEL.output_dim)
            feature_dim = args.MODEL.output_dim
        elif args.MODEL.network == "resnet50":
            net = network.resnet.resnet50(pretrained=False)
            feature_dim = 2048
        # net = network.test_model.Model()
        # feature_dim = 512

        if args.TRAIN.vae:
            # vae = network.vae.VAE(feature_dim, args.MODEL.vae_zdim, args.MODEL.vae_layers)
            vae = network.vae.VariationalModule(
                feature_dim, args.MODEL.vae_zdim, args.MODEL.vae_layers)
            msg = "   VAE   "
            msglogger.info("#"*15+msg + "#"*15)
            msglogger.info(str(vae))
            msglogger.info("#"*(30+len(msg)))

            feature_dim = args.MODEL.vae_zdim

        if len(args.MODEL.linear_layers):
            head = network.head.MLP(
                feature_dim, args.num_classes, args.MODEL.linear_layers)
        elif args.TRAIN.self_supervised_method == "supervise_cossim":
            head = network.cosinehead.CosineMarginProduct(
                in_feature=feature_dim, out_feature=args.num_classes, s=args.TRAIN.logit_scale
            )
        else:
            head = network.head.Head(feature_dim, args.num_classes)
        msg = "   HEAD   "
        msglogger.info("#"*15+msg + "#"*15)
        msglogger.info(str(head))
        msglogger.info("#"*(30+len(msg)))

    if args.MODEL.resume and not args.TRAIN.vae:
        net, start_epoch = network.model_io.load_model(
            net, args.MODEL.resume_net_path, logger=msglogger)
        args.TRAIN.start_epoch = start_epoch

    if args.TRAIN.vae:
        net, _ = network.model_io.load_model(
            net, args.MODEL.resume_extractor_path, logger=msglogger, freeze=args.TRAIN.vae_pretrain_freeze
        )

    if args.TRAIN.self_supervised_method == "simclr" and not args.TRAIN.finetune_linear:

        if args.TRAIN.vae:
            # criterion = lossfunction.simclr_loss.SimCLRLossV(args.DATA.batch_size, device=args.device, feature_dim=args.MODEL.vae_zdim)
            criterion = lossfunction.simclr_loss.SimCLRLoss(
                args.DATA.batch_size, scale=args.TRAIN.logit_scale, device=args.device)
            rec_loss = lossfunction.metric.mseloss
            if args.TRAIN.prior_agg:
                kl_loss = lossfunction.kl_div.kl_div_normal_2
            else:
                kl_loss = lossfunction.kl_div.kl_div_normal
        else:
            criterion = lossfunction.simclr_loss.SimCLRLoss(
                args.DATA.batch_size, scale=args.TRAIN.logit_scale, device=args.device)
    else:
        criterion = nn.CrossEntropyLoss()

    trn_embedder, val_embedder = embeddings.meta.get_embedder(args)

    if args.MODEL.embedding_algorithm == "nca" and not args.MODEL.is_instanciate_each_iter:
        args.MODEL.trn_embedder = trn_embedder
        args.MODEL.val_embedder = val_embedder

    if args.TRAIN.finetune_linear:
        wrapper = network.wrapper.LossWrapLinear(args, net, criterion)
    else:
        if args.TRAIN.self_supervised_method == "simclr":
            if args.TRAIN.vae:
                wrapper = network.wrapper.LossWrapVAE(
                    args, net, vae, head, rec_loss=rec_loss, kl_loss=kl_loss, cont_loss=criterion)
            else:
                wrapper = network.wrapper.LossWrapSimCLR(
                    args, net, head, criterion)
        else:
            if args.DATA.is_episode:
                if "lda" in args.MODEL.embedding_algorithm and args.MODEL.embedding_flag:
                    epoch_func = epoch_func_integrate
                    wrapper = network.wrapper.LossWrapEpisodeLDA(
                        args, net, head, criterion, trn_embedding=trn_embedder, val_embedding=val_embedder)

                elif "nca" in args.MODEL.embedding_algorithm and args.MODEL.embedding_flag:
                    epoch_func = epoch_func_integrate
                    wrapper = network.wrapper.LossWrapEpisodeNCA(
                        args, net, head, criterion, trn_embedding=trn_embedder, val_embedding=val_embedder)

                else:
                    epoch_func = epoch_func_normal
                    wrapper = network.wrapper.LossWrapEpisode(
                        args, net, head, criterion, trn_embedding=trn_embedder, val_embedding=val_embedder)
            else:
                if "nca" in args.MODEL.embedding_algorithm and args.MODEL.embedding_flag:
                    epoch_func = epoch_func_integrate
                    wrapper = network.wrapper.LossWrapNCA(args, net, head, criterion, trn_embedding=trn_embedder, val_embedding=val_embedder)

                else:
                    wrapper = network.wrapper.LossWrap(args, net, head, criterion)
                    epoch_func = epoch_func_integrate

    wrapper = wrapper.cuda()

    if args.run_mode == "test":
        epoch = -1
        # if args.DATA.is_episode:
            
        # else:
        val_result = epoch_func.valid_epoch(
                    wrapper, val_loader, epoch, args, logger=val_logger)
        # pass                msg = "Valid: "
        msg = "Valid:"
        for name, value in val_result.items():
            if name.startswith("each"):
                continue
            msg += "{}:{:.4f} ".format(name, value)
        msglogger.info(msg)

    elif args.run_mode == "train":

        if args.OPTIM.optimizer == "adam":
            if args.TRAIN.vae:
                if args.TRAIN.vae_pretrain_freeze:
                    parameters = []
                else:
                    parameters = [
                        {"params": net.parameters(), "lr": args.OPTIM.lr,
                         "weight_deacy": 1e-6},
                    ]
                parameters.append(
                    {"params": vae.parameters(), "lr": args.OPTIM.lr,
                     "weight_decay": 1e-6}
                )
            else:
                parameters = [
                    {"params": net.parameters(), "lr": args.OPTIM.lr,
                     "weight_deacy": 1e-6},
                ]
            if not args.TRAIN.finetune_linear:
                parameters.append(
                    {"params": head.parameters(), "lr": args.OPTIM.lr,
                     "weight_deacy": 1e-6},
                )

            if hasattr(args.MODEL, "trn_embedder"):
                parameters.append({
                    "params": args.MODEL.trn_embedder.nca.parameters(), 
                    "lr": args.OPTIM.lr,
                    "weight_deacy": 1e-6,
                })

            optimizer = torch.optim.Adam(
                parameters
            )
        elif args.TRAIN.optimizer == "sgd":
            if args.TRAIN.vae:
                parameters = []
                parameters.append(
                    {"params": vae.parameters(), "lr": args.OPTIM.lr,
                     "weight_decay": 1e-6, "momentum": 0.9}
                )
            else:
                parameters = [
                    {"params": net.parameters(), "lr": args.OPTIM.lr,
                     "weight_deacy": 1e-6, "momentum": 0.9},
                ]
            if not args.TRAIN.finetune_linear:
                parameters.append(
                    {"params": head.parameters(), "lr": args.OPTIM.lr,
                     "weight_deacy": 1e-6, "momentum": 0.9},
                )
            if hasattr(args.MODEL, "trn_embedder"):
                parameters.append({
                    "params": args.MODEL.trn_embedder.nca.parameters(), 
                    "lr": args.OPTIM.lr,
                    "weight_deacy": 1e-6,
                })

            optimizer = torch.optim.SGD(
                parameters
            )
        else:
            raise NotImplementedError

        if args.TRAIN.fp16:
            opt_level = "O1"
            (wrapper,), (optimizerr) = amp.initialize(
                [wrapper], [optimizer], opt_level=opt_level)

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
        train_since = time.time()
        for epoch in range(args.TRAIN.start_epoch, args.TRAIN.total_epoch+1):
            msglogger.info("Start epoch {}".format(epoch))
            trn_info = epoch_func.train_epoch(
                wrapper, train_loader, optimizer, epoch, args, logger=trn_logger)

            if args.TRAIN.self_supervised_method.startswith("supervise"):
                # if args.DATA.is_episode:
                val_info = epoch_func.valid_epoch(
                    wrapper, val_loader, epoch, args, logger=val_logger)
                # else:
                    # val_info = None
                # val_info = None
            else:
                val_info = epoch_func.valid_epoch(
                    wrapper, val_loader, epoch, args, logger=val_logger)
            # trn_info = {"acc":0}
            # val_info = {"acc": 0}
            val_result = val_info
            # print(val_result)
            # sfda
            if not args.TRAIN.finetune_linear and val_info is not None:
                # if epoch % args.DATA.feature_save_freq == 0 and epoch > 0:
                if args.TRAIN.vae:
                    val_result = epoch_func.valid_inference_vae(
                        wrapper.model, wrapper.vae, val_loader, epoch, args, logger=val_logger)
                # elif not args.DATA.is_episode:
                elif val_info is None:
                    val_result = epoch_func.valid_inference(
                        wrapper.model, val_loader, epoch, args, logger=val_logger)
            # val_result = epoch_func.valid_inference(
            #     wrapper.model, train_loader, val_loader, args, val_logger)
            for param_group in optimizer.param_groups:
                lr = param_group["lr"]
                break

            # score = val_result["linear_acc"]
            if args.TRAIN.finetune_linear:
                score = val_info["acc"]
                val_msg = "Test: "
                for name, value in val_result.items():
                    val_msg += "{}:{:.4f} ".format(name, value)
                val_msg_2 = ""
            elif args.TEST.mode == "knn_eval":
                # score = val_info["pseudo_acc"]
                score = val_result["top1"]
                val_msg = "Test: "
                for name, value in val_result.items():
                    val_msg += "{}:{:.4f} ".format(name, value)
                val_msg_2 = ""

            # elif args.TEST.mode == "few_shot" and not args.DATA.is_episode:
            elif args.TEST.mode == "few_shot":
                if "lda" in str(trn_embedder):
                    score = val_result["mean_m_acc_{}_l2euc_lda_2".format(
                        args.TEST.n_support)]

                if "nca" in str(trn_embedder).lower():
                    score = val_result["mean_m_acc_{}_l2euc_nca_dim2-iter05".format(
                        args.TEST.n_support)]

                else:
                    score = val_result["mean_m_acc_{}_cossim".format(
                        args.TEST.n_support)]
                val_msg = "Test: "
                for name, value in val_result.items():
                    if name.startswith("mean"):
                        val_msg += "{}:{:.4f} ".format(name, value)
                val_msg += "\n"
                key_list = sorted(list(val_result.keys()))
                val_msg_2 = ""
                for name in key_list:
                    if name.startswith("each_"):
                        val_msg_2 += "{}:{:.4f} ".format(name, val_result[name])
            # else:
                # score = val_info["pseudo_acc"]
                # val_msg = ""

            iter_end = time.time() - train_since
            msg = "Epoch:[{}/{}] lr:{} elapsed_time:{:.4f}s mean epoch time:{:.4f}s".format(epoch,
                                                                                            args.TRAIN.total_epoch, lr, iter_end, iter_end/epoch)
            msglogger.info(msg)
            msglogger.info(val_msg)
            val_logger.debug(msg)
            val_logger.debug(val_msg_2)

            if val_info is not None:
                msg = "Valid: "
                for name, value in val_info.items():
                    if name.startswith("mean") or name.startswith("each"):
                        continue
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
            
            if args.MODEL.embedding_algorithm == "nca" and not args.MODEL.is_instanciate_each_iter:
                embedder_info = {
                    "trn": trn_embedder.nca.state_dict(),
                    "val": val_embedder.nca.state_dict()
                }

            else:
                embedder_info = None

            network.model_io.save_model(wrapper, optimizer, val_info, is_best, epoch,
                                        logger=msglogger, multi_gpus=args.multi_gpus,
                                        model_save_dir=args.MODEL.save_dir, delete_old=args.MODEL.delete_old,
                                        fp16_train=args.TRAIN.fp16, amp=amp,
                                        embedder=embedder_info)
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
