import logging

import torch
from torch import nn, optim

import utils
from loss_kd import *

class GKTClientTrainer(object):
    def __init__(self, client_index, local_training_data, local_test_data, local_sample_number_train, local_sample_number_test, device,
                 client_model, args):
        self.client_index = client_index
        self.local_training_data = local_training_data[client_index]
        self.local_test_data = local_test_data[client_index]

        self.local_sample_number_train = local_sample_number_train
        self.local_sample_number_test = local_sample_number_test

        self.args = args

        self.device = device
        self.client_model = client_model

        logging.info("client device = " + str(self.device))
        self.client_model.to(self.device)

        self.model_params = self.master_params = self.client_model.parameters()

        optim_params = utils.bnwd_optim_params(self.client_model, self.model_params,
                                                            self.master_params) if args.no_bn_wd else self.master_params

        if self.args.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(optim_params, lr=self.args.lr, momentum=0.9,
                                             nesterov=True,
                                             weight_decay=self.args.wd)
        elif self.args.optimizer == "Adam":
            self.optimizer = optim.Adam(optim_params, lr=self.args.lr, weight_decay=0.0001, amsgrad=True)

        self.criterion_CE = nn.CrossEntropyLoss()
        self.criterion_KL = utils.KL_Loss(self.args.temperature)

        self.server_logits_dict = dict()

    def get_sample_number_train(self):
        return self.local_sample_number_train

    def get_sample_number_test(self):
        return self.local_sample_number_test

    def update_large_model_logits(self, logits):
        self.server_logits_dict = logits

    def train(self):
        extracted_feature_dict = dict()
        logits_dict = dict()
        labels_dict = dict()
        extracted_feature_dict_test = dict()
        labels_dict_test = dict()

        if self.args.whether_training_on_client == 1:
            self.client_model.train()
            epoch_loss = []
            for epoch in range(self.args.epochs_client):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.local_training_data):
                    labels=torch.tensor(labels, dtype=torch.long)
                    images, labels = images.to(self.device), labels.to(self.device)

                    log_probs, _ = self.client_model(images)
                    loss_true = self.criterion_CE(log_probs, labels)
                    if len(self.server_logits_dict) != 0:
                        large_model_logits = torch.from_numpy(self.server_logits_dict[batch_idx]).to(
                            self.device)
                        KD_Loss= get_loss_kd_client(self.args.method,self.args,self.client_index)
                        loss_kd = KD_Loss(log_probs, large_model_logits)
                        loss = loss_true + self.args.alpha * loss_kd
                    else:
                        loss = loss_true
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    logging.info('client {} - Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        self.client_index, epoch, batch_idx * len(images), len(self.local_training_data.dataset),
                                                  100. * batch_idx / len(self.local_training_data), loss.item()))
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

        self.client_model.eval()
        metrics=self.eval_on_the_client()

        for batch_idx, (images, labels) in enumerate(self.local_training_data):
            images, labels = images.to(self.device), labels.to(self.device)
            log_probs, extracted_features = self.client_model(images)
            extracted_feature_dict[batch_idx] = extracted_features.cpu().detach().numpy()
            log_probs = log_probs.cpu().detach().numpy()
            logits_dict[batch_idx] = log_probs
            labels_dict[batch_idx] = labels.cpu().detach().numpy()

        for batch_idx, (images, labels) in enumerate(self.local_test_data):
            test_images, test_labels = images.to(self.device), labels.to(self.device)
            _, extracted_features_test = self.client_model(test_images)
            extracted_feature_dict_test[batch_idx] = extracted_features_test.cpu().detach().numpy()
            labels_dict_test[batch_idx] = test_labels.cpu().detach().numpy()

        return extracted_feature_dict, logits_dict, labels_dict, extracted_feature_dict_test, labels_dict_test,metrics

    def eval_on_the_client(self):
        # set model to evaluation mode
        self.model_global=self.client_model
        self.model_global.eval()
        

        for batch_idx, (images, labels) in enumerate(self.local_test_data):
            loss_avg = utils.RunningAverage()
            accTop1_avg = utils.RunningAverage()
            accTop5_avg = utils.RunningAverage()
            images, labels = images.to(self.device), labels.to(self.device)
            labels=torch.tensor(labels,dtype=torch.long)
            log_probs, extracted_features = self.client_model(images)
            loss = self.criterion_CE(log_probs, labels)
            metrics = utils.accuracy(log_probs, labels, topk=(1, 5))
            accTop1_avg.update(metrics[0].item())
            accTop5_avg.update(metrics[1].item())
            loss_avg.update(loss.item())
        print(loss_avg,type(loss_avg))

        test_metrics = {str(self.client_index)+' test_loss': loss_avg.value(),
                        str(self.client_index)+' test_accTop1': accTop1_avg.value(),
                        str(self.client_index)+' test_accTop5': accTop5_avg.value(),
                        }
        print({str(self.client_index)+" Test/Loss": test_metrics[str(self.client_index)+' test_loss']})
        print({str(self.client_index)+" Test/AccTop1": test_metrics[str(self.client_index)+' test_accTop1']})
        return test_metrics