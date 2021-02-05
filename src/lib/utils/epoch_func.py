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
import lib.evaluation.few_shot_eval as few_shot_eval
from tqdm import tqdm
from sklearn import linear_model
try:
    from apex import amp
except ImportError:
    pass


def check_equal(a, b):
    assert a == b, (a, b)

def make_directory(path):
    if os.path.exists(str(path)) is False:
        os.makedirs(str(path))


def train_epoch(wrappered_model, train_loader, optimizer, epoch, args, logger=None):
    wrappered_model.train()
    if args.TRAIN.vae:
        wrappered_model.model.eval()
    else:
        wrappered_model.model.train()

    meter = average_meter.AverageMeter()
    # train_loader.dataset.set_eval()
    # else:
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
            input_x, pseudo_label = simclr_utils.setup_simclr(
                input_x, data["data2"])

        elif args.TRAIN.self_supervised_method == "rotate" or args.TRAIN.finetune_linear:
            pseudo_label = data["label"]
        elif args.TRAIN.vae and not args.TRAIN.startswith("supervise"):
            pseudo_label = None
        elif args.TRAIN.self_supervised_method.startswith("supervise"):
            pseudo_label = data["label"]
        since = time.time()
        output = wrappered_model(input_x, pseudo_label)

        if len(output) == 2:
            loss, output = output
        elif len(output) == 3:
            if args.DATA.is_episode:
                loss, output, meta_label = output
                pseudo_label = meta_label.cpu()
            else:
                kl_loss, cont_loss, output = output
                loss = kl_loss + cont_loss
                meter.add_value('kl_loss', kl_loss)
                meter.add_value('cont_loss', cont_loss)
        elif len(output) == 4:
            if args.DATA.is_episode:
                loss, output, meta_label, raw_logits = output
                B, _, _, D = raw_logits.shape
                raw_logits = raw_logits.reshape(-1, D)
                pseudo_label = meta_label.cpu()
            else:
                rec_loss, kl_loss, cont_loss, output = output
                loss = rec_loss + kl_loss + cont_loss
                meter.add_value('rec_loss', rec_loss)
                meter.add_value('kl_loss', kl_loss)
                meter.add_value('cont_loss', cont_loss)
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
            if batch_idx >= 5:
                break
        since = time.time()
    # all_logits = np.concatenate(all_logits)
    train_info = meter.get_summary()

    return train_info


def valid_epoch(wrappered_model, train_loader, epoch, args, logger=None):
    wrappered_model.eval()

    meter = average_meter.AverageMeter()
    if args.TRAIN.self_supervised_method.startswith("supervise"):
        train_loader.dataset.set_eval()
    else:
        train_loader.dataset.set_train()
    iter_num = len(train_loader)
    iter_since = time.time()
    since = iter_since

    with torch.no_grad():
        save_num = 64
        now_num = 0
        infer_data = []
        train_logits = []
        train_labels = []
        # all_logits = []
        for batch_idx, data in tqdm(enumerate(train_loader), total=iter_num, desc="validation"):

            meter.add_value("time_data", time.time() - since)
            input_x = data["data"]
            if args.TRAIN.self_supervised_method == "simclr" and not args.TRAIN.finetune_linear:
                input_x, pseudo_label = simclr_utils.setup_simclr(
                    input_x, data["data2"])

            elif args.TRAIN.self_supervised_method == "rotate" or args.TRAIN.finetune_linear:
                pseudo_label = data["label"]
            elif args.TRAIN.vae and not args.TRAIN.startswith("supervise"):
                pseudo_label = None
            elif args.TRAIN.self_supervised_method.startswith("supervise"):
                pseudo_label = data["label"]
                label = data["label"].reshape(-1)
            since = time.time()
            # print("input x:", input_x.shape)
            output = wrappered_model(input_x, pseudo_label)

            if len(output) == 2:
                loss, output = output
            elif len(output) == 3:
                if args.DATA.is_episode:
                    loss, output, meta_label = output
                    pseudo_label = meta_label.cpu()
                else:
                    kl_loss, cont_loss, output = output
                    loss = kl_loss + cont_loss
                    meter.add_value('kl_loss', kl_loss)
                    meter.add_value('cont_loss', cont_loss)
            elif len(output) == 4:
                if args.DATA.is_episode:
                    loss, output, meta_label, raw_logits = output
                    B, _, _, D = raw_logits.shape
                    pseudo_label = meta_label.cpu()
                else:
                    rec_loss, kl_loss, cont_loss, output = output
                    loss = rec_loss + kl_loss + cont_loss
                    meter.add_value('rec_loss', rec_loss)
                    meter.add_value('kl_loss', kl_loss)
                    meter.add_value('cont_loss', cont_loss)
            meter.add_value("time_f", time.time()-since)

            _, pred = torch.max(output, dim=1)
            correct = np.mean(pred.cpu().numpy() == pseudo_label.numpy())

            meter.add_value("time_f", time.time()-since)
            _, pred = torch.max(output, dim=1)
            correct = np.mean(pred.cpu().numpy() == pseudo_label.numpy())
            if args.TRAIN.finetune_linear:
                meter.add_value("acc", correct)
            else:
                meter.add_value("pseudo_acc", correct)

            if args.TRAIN.self_supervised_method.startswith("supervise"):
                raw_logits = raw_logits.cpu()
                # print(raw_logits.shape)
                # for n_sp in [1, args.TEST.n_support]:
                #     for meta_mode in ("cossim", "euc", "innerprod"):
                #         acc = few_shot_eval.fewshot_eval_meta(
                #             raw_logits=raw_logits, n_support=n_sp,
                #             n_query=args.TEST.n_query,
                #             meta_mode=meta_mode
                #         )
                #         meter.add_value(
                #             "meta_mean_{}_{}".format(n_sp, meta_mode), acc)
                train_labels.append(label)
                raw_logits = raw_logits.reshape(-1, D)
                check_equal(len(raw_logits), len(label))
                train_logits.append(raw_logits)
            meter.add_value("loss_total", loss)

            # if now_num < save_num:
            # infer_data.append(input_x)
            now_num += save_num

            if args.debug:
                if batch_idx >= 5:
                    break
            since = time.time()

        if args.TRAIN.self_supervised_method.startswith("supervise"):
            train_labels = torch.cat(train_labels)
            train_logits = torch.cat(train_logits)
            result = few_shot_eval.evaluate_fewshot(
                train_logits.numpy(), train_labels.numpy(), args)
        # infer_data = torch.cat(infer_data, dim=0)[:save_num]
        # all_logits = np.concatenate(all_logits)
    train_info = meter.get_summary()
    train_info.update(result)
    # inference(wrappered_model.model, infer_data, args, epoch)

    return train_info


def valid_inference_vae(net, vae, base_loader, epoch, args, val_loader=None, logger=None):
    base_loader.dataset.set_eval()
    if val_loader is not None:
        val_loader.dataset.set_eval()
    net.eval()
    vae.eval()
    with torch.no_grad():
        logger.info("Inference data ... ")
        iter_num = len(base_loader)
        train_labels = []
        train_logits = {}
        # base_loader.drop_last = False
        for idx, data in tqdm(enumerate(base_loader), total=iter_num, desc="train"):
            input_x = data["data"]
            label = data["real_label"]
            input_x = input_x.cuda()
            raw_logits = net(input_x)
            mean_v, sigma_v = vae.encoder(raw_logits)
            train_labels.append(label)
            try:
                train_logits["mean"].append(mean_v.cpu())
                train_logits["sigma"].append(sigma_v.cpu())
            except KeyError:
                train_logits["mean"] = [mean_v.cpu()]
                train_logits["sigma"] = [sigma_v.cpu()]
            if args.debug:
                if idx >= 3:
                    break

        train_labels = torch.cat(train_labels)
        for key, value in train_logits.items():
            train_logits[key] = torch.cat(value).numpy()

        if args.TRAIN.vae:
            result = few_shot_eval.evaluate_fewshot_vae(
                train_logits["mean"], train_logits["sigma"], train_labels.numpy(), args)
        else:
            result = few_shot_eval.evaluate_fewshot(
                train_logits["mean"], train_labels.numpy(), args)
    return result


def valid_inference(net, base_loader, epoch, args, val_loader=None, logger=None):
    base_loader.dataset.set_eval()
    if val_loader is not None:
        val_loader.dataset.set_eval()
    net.eval()
    with torch.no_grad():
        logger.info("Inference data ... ")
        iter_num = len(base_loader)
        train_labels = []
        train_logits = []
        # base_loader.drop_last = False
        for idx, data in tqdm(enumerate(base_loader), total=iter_num, desc="train"):
            input_x = data["data"]
            label = data["real_label"]
            input_x = input_x.cuda()
            raw_logits = net(input_x)
            train_labels.append(label)
            train_logits.append(raw_logits.cpu())
            if args.debug:
                if idx >= 3:
                    break

        train_labels = torch.cat(train_labels)
        train_logits = torch.cat(train_logits)

        # output_trn_feats = {
        #     "label": train_labels,
        #     "feature": train_logits
        # }
        if val_loader is not None:
            iter_num = len(val_loader)
            valid_labels = []
            valid_logits = []
            for idx, data in tqdm(enumerate(val_loader), total=iter_num, desc="valid"):
                label = data["real_label"]
                input_x = data["data"]
                input_x = input_x.cuda()
                raw_logits = net(input_x)
                valid_labels.append(label)
                valid_logits.append(raw_logits.cpu())
                if args.debug:
                    if idx >= 3:
                        break
            valid_labels = torch.cat(valid_labels)
            valid_logits = torch.cat(valid_logits)
            assert args.has_same is False
        else:
            assert args.has_same
            valid_logits = train_logits
            valid_labels = train_labels
        # train_loader.drop_last = True

        # output_val_feats = {
        #     "label": valid_labels,
        #     "feature": valid_logits
        # }

    # save_feature_dir = os.path.join(args.DATA.feature_root_dir, "epoch_{:0>5}".format(epoch))
    # make_directory(save_feature_dir)
    # save_feature_path = os.path.join(save_feature_dir, "train.pickle")
    # with open(save_feature_path, "wb") as pkl:
    #     pickle.dump(output_trn_feats, pkl)
    # save_feature_path = os.path.join(save_feature_dir, "valid.pickle")
    # with open(save_feature_path, "wb") as pkl:
    #     pickle.dump(output_val_feats, pkl)
    # return output_val_feats
        if args.TEST.mode == "knn_eval":
            result = knn_eval(train_logits, train_labels,
                              valid_logits, valid_labels, args, logger)
        elif args.TEST.mode == "few_shot":
            result = few_shot_eval.evaluate_fewshot(
                train_logits.numpy(), train_labels.numpy(), args)
    return result


def knn_eval(x_data, y_data, x_test, y_test, args, logger, has_same=False):
    logger.info(f"start knn(k={args.TEST.neighbor_k}) ... ")
    top1_acc = 0
    top5_acc = 0
    num_classes = y_data.max() + 1
    x_data = x_data / \
        torch.sqrt(torch.sum(x_data * x_data, axis=-1, keepdim=True))
    # x_test = x_data
    x_test = x_test / \
        torch.sqrt(torch.sum(x_test * x_test, axis=-1, keepdim=True))
    BT = x_test.size(0)
    batch_num = (BT+(args.TEST.batch_size-1)) // args.TEST.batch_size
    num = 0
    for batch_idx in range(batch_num):
        batch_test = x_test[batch_idx *
                            args.TEST.batch_size: (batch_idx+1)*args.TEST.batch_size]
        batch_test_l = y_data[batch_idx *
                              args.TEST.batch_size: (batch_idx+1)*args.TEST.batch_size]
        bsize = batch_test.size(0)
        num += bsize
        sim_matrix = torch.mm(batch_test, x_data.T)
        sim_weight, sim_indices = sim_matrix.topk(
            k=args.TEST.neighbor_k+args.has_same, dim=-1)
        # print("before:", args.TEST.neighbor_k, sim_indices.size(), args.has_same)
        # print(sim_weight[:30, :2], sim_indices[:30, :2])
        sim_weight = sim_weight[:, int(args.has_same):]
        sim_indices = sim_indices[:, int(args.has_same):]
        # print("after", args.TEST.neighbor_k, sim_indices.size(), args.has_same)
        # print(sim_weight[:30, :2], sim_indices[:30, :2])
        # knjdfklsgjkl
        # [B, K]
        sim_labels = torch.gather(y_data.expand(
            bsize, -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight * args.TEST.test_logit_scale).exp()
        # print("sim label:", sim_labels.size())

        # counts for each class
        one_hot_label = torch.zeros(
            bsize * args.TEST.neighbor_k, num_classes, device=sim_labels.device)
        # print("one hot label:", one_hot_label.size())
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(
            dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(
            bsize, -1, num_classes) * sim_weight.unsqueeze(dim=-1), dim=1)

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        top1_acc += torch.sum((pred_labels[:, :1] == batch_test_l.unsqueeze(
            dim=-1)).any(dim=-1).float()).item()
        top5_acc += torch.sum((pred_labels[:, :5] == batch_test_l.unsqueeze(
            dim=-1)).any(dim=-1).float()).item()
    # print(num, BT, batch_num, BT+(args.TEST.batch_size-1), args.TEST.batch_size)
    top1_acc /= BT
    top5_acc /= BT
    result = {
        "top1": top1_acc,
        "top5": top5_acc
    }
    # torch.gather(x_data, dim=1, index=indice)

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
