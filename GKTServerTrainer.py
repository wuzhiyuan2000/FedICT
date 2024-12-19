import logging
import os
import shutil

import torch
#import wandb
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import utils
from loss_kd import *


class GKTServerTrainer(object):
    def __init__(self, client_num, device, server_model, args):
        self.client_num = client_num
        self.device = device
        self.args = args
        """
            when use data parallel, we should increase the batch size accordingly (single GPU = 64; 4 GPUs = 256)
            One epoch training time: single GPU (64) = 1:03; 4 x GPUs (256) = 38s; 4 x GPUs (64) = 1:00
            Note that if we keep the same batch size, the frequent GPU-CPU-GPU communication will lead to
            slower training than a single GPU.
        """
        # server model
        self.model_global = server_model

        if args.multi_gpu_server and torch.cuda.device_count() > 1:
            self.model_global = nn.DataParallel(self.model_global, device_ids=[0, 1, 2, 3]).to(device)

        self.model_global.train()
        self.model_global.to(self.device)

        self.model_params = self.master_params = self.model_global.parameters()

        optim_params = utils.bnwd_optim_params(self.model_global, self.model_params,
                                                            self.master_params) if args.no_bn_wd else self.master_params

        if self.args.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(optim_params, lr=self.args.lr, momentum=0.9,
                                             nesterov=True,
                                             weight_decay=self.args.wd)
        elif self.args.optimizer == "Adam":
            self.optimizer = optim.Adam(optim_params, lr=self.args.lr, weight_decay=0.0001, amsgrad=True)

        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max')

        self.criterion_CE = nn.CrossEntropyLoss()
        self.criterion_KL = utils.KL_Loss(self.args.temperature)
        self.criterion_DKC_KL=utils.DKC_KL_Loss_search_based(self.args.temperature,self.args.target_ent,self.args.class_num)#utils.DKC_KL_Loss(self.args.temperature,self.args.T,self.args.S,self.args.class_num)
        self.best_acc = 0.0

        # key: client_index; value: extracted_feature_dict
        self.client_extracted_feauture_dict = dict()

        # key: client_index; value: logits_dict
        self.client_logits_dict = dict()

        # key: client_index; value: labels_dict
        self.client_labels_dict = dict()

        # key: client_index; value: labels_dict
        self.server_logits_dict = dict()

        # for test
        self.client_extracted_feauture_dict_test = dict()
        self.client_labels_dict_test = dict()

        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.train_acc_dict = dict()
        self.train_loss_dict = dict()
        self.test_acc_avg = 0.0
        self.test_loss_avg = 0.0

        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def add_local_trained_result(self, index, extracted_feature_dict, logits_dict, labels_dict,
                                 extracted_feature_dict_test, labels_dict_test):
        logging.info("add_model. index = %d" % index)
        self.client_extracted_feauture_dict[index] = extracted_feature_dict
        self.client_logits_dict[index] = logits_dict
        self.client_labels_dict[index] = labels_dict
        self.client_extracted_feauture_dict_test[index] = extracted_feature_dict_test
        self.client_labels_dict_test[index] = labels_dict_test

        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.client_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def get_global_logits(self, client_index):
        return self.server_logits_dict[client_index]

    def train(self, round_idx):
        if self.args.sweep == 1:
            self.sweep(round_idx)
        else:
            if self.args.whether_training_on_client == 1:
                self.train_and_distill_on_client(round_idx)
            else:
                self.do_not_train_on_client(round_idx)

    def get_server_epoch_strategy_test(self):
        return 1, True

    # ResNet56
    def get_server_epoch_strategy(self, round_idx):
        whether_distill_back = True
        # set the training strategy
        epochs = self.args.epochs_server
        return epochs, whether_distill_back


    def train_and_distill_on_client(self, round_idx):
        epochs_server, whether_distill_back = self.get_server_epoch_strategy(round_idx)

        # train according to the logits from the client
        self.train_and_eval(round_idx, epochs_server)

        # adjust the learning rate based on the number of epochs.
        # https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
        self.scheduler.step(self.best_acc, epoch=round_idx)

    def do_not_train_on_client(self, round_idx):
        raise Exception("invalid function called")
        self.train_and_eval(round_idx, 1)
        self.scheduler.step(self.best_acc, epoch=round_idx)

    def sweep(self, round_idx):
        # train according to the logits from the client
        self.train_and_eval(round_idx, self.args.epochs_server)
        self.scheduler.step(self.best_acc, epoch=round_idx)

    def train_and_eval(self, round_idx, epochs):
        print("train_and_eval start successfully!")
        for epoch in range(epochs):
            logging.info("train_and_eval. round_idx = %d, epoch = %d" % (round_idx, epoch))
            
            train_metrics=None
            if len(self.client_extracted_feauture_dict)!=0:
                train_metrics = self.train_large_model_on_the_server()
            else:
                train_metrics={'train_loss': 100,
                         'train_accTop1': 0.0,
                         'train_accTop5': 0.0}

    def train_large_model_on_the_server(self):
        print("start train large model on the server")
        # clear the server side logits
        for key in self.server_logits_dict.keys():
            self.server_logits_dict[key].clear()
        self.server_logits_dict.clear()

        self.model_global.train()

        loss_avg = utils.RunningAverage()
        accTop1_avg = utils.RunningAverage()
        accTop5_avg = utils.RunningAverage()

        print("start distill on server")

        for client_index in self.client_extracted_feauture_dict.keys():
            extracted_feature_dict = self.client_extracted_feauture_dict[client_index]
            logits_dict = self.client_logits_dict[client_index]
            labels_dict = self.client_labels_dict[client_index]

            s_logits_dict = dict()
            self.server_logits_dict[client_index] = s_logits_dict
            for batch_index in extracted_feature_dict.keys():
                batch_feature_map_x = torch.from_numpy(extracted_feature_dict[batch_index]).to(self.device)
                batch_logits = torch.from_numpy(logits_dict[batch_index]).float().to(self.device)
                batch_labels = torch.from_numpy(labels_dict[batch_index]).long().to(self.device)

                output_batch = self.model_global(batch_feature_map_x)

                if self.args.whether_distill_on_the_server == 1:
                    KD_loss = get_loss_kd_server(self.args.method,self.args,client_index)
                    loss_kd = KD_loss(output_batch.clone(), batch_logits.clone()).to(self.device)
                    loss_true = self.criterion_CE(output_batch, batch_labels).to(self.device)
                    loss = loss_kd + self.args.alpha * loss_true
                else:
                    loss_true = self.criterion_CE(output_batch, batch_labels).to(self.device)
                    loss = loss_true

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                metrics = utils.accuracy(output_batch, batch_labels, topk=(1, 5))
                accTop1_avg.update(metrics[0].item())
                accTop5_avg.update(metrics[1].item())
                loss_avg.update(loss.item())
                s_logits_dict[batch_index] = output_batch.cpu().detach().numpy()
        train_metrics = {'train_loss': loss_avg.value(),
                         'train_accTop1': accTop1_avg.value(),
                         'train_accTop5': accTop5_avg.value()}

        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in train_metrics.items())
        logging.info("- Train metrics: " + metrics_string)
        return train_metrics

    def eval_large_model_on_the_server(self):
        self.model_global.eval()
        loss_avg = utils.RunningAverage()
        accTop1_avg = utils.RunningAverage()
        accTop5_avg = utils.RunningAverage()
        with torch.no_grad():
            for client_index in self.client_extracted_feauture_dict_test.keys():
                extracted_feature_dict = self.client_extracted_feauture_dict_test[client_index]
                labels_dict = self.client_labels_dict_test[client_index]

                for batch_index in extracted_feature_dict.keys():
                    batch_feature_map_x = torch.from_numpy(extracted_feature_dict[batch_index]).to(self.device)
                    batch_labels = torch.from_numpy(labels_dict[batch_index]).long().to(self.device)

                    output_batch = self.model_global(batch_feature_map_x)
                    loss = self.criterion_CE(output_batch, batch_labels)
                    metrics = utils.accuracy(output_batch, batch_labels, topk=(1, 5))
                    accTop1_avg.update(metrics[0].item())
                    accTop5_avg.update(metrics[1].item())
                    loss_avg.update(loss.item())
        test_metrics = {'test_loss': loss_avg.value(),
                        'test_accTop1': accTop1_avg.value(),
                        'test_accTop5': accTop5_avg.value()}

        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in test_metrics.items())
        logging.info("- Test  metrics: " + metrics_string)
        return test_metrics
