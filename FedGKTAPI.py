from GKTServerTrainer import GKTServerTrainer
from GKTClientTrainer import GKTClientTrainer
import torch
import os
import numpy as np
import sys


class FedGKT_standalone_API:
    def __init__(self,server_model,client_models, train_data_local_num_dict, test_data_local_num_dict,
                             train_data_local_dict, test_data_local_dict, args):
        self.server_trainer=GKTServerTrainer(args.client_number,torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),server_model,args)
        self.client_trainers=[GKTClientTrainer(i, train_data_local_dict, test_data_local_dict, train_data_local_num_dict, test_data_local_num_dict,
                               torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), client_models[i], args) for i in range(args.client_number)]
        self.global_extracted_feature_dict=dict()
        self.global_logits_dict=dict()
        self.global_labels_dict=dict()
        self.global_extracted_feature_dict_test=dict()
        self.global_labels_dict_test=dict()
    

    def do_fedgkt_stand_alone(self,server_model,client_models, train_data_local_num_dict, test_data_local_num_dict,
                             train_data_local_dict, test_data_local_dict, args):
    
        for global_epoch in range(1000000):
            metrics_all={'test_loss':[],'test_accTop1':[],'test_accTop5':[]}   
            for client_index,trainer in enumerate(self.client_trainers):
                if len(self.server_trainer.server_logits_dict)!=0:
                    trainer.server_logits_dict=self.server_trainer.server_logits_dict[trainer.client_index]
                extracted_feature_dict, logits_dict, labels_dict, extracted_feature_dict_test, labels_dict_test,metrics =trainer.train()
                self.global_extracted_feature_dict[trainer.client_index]=extracted_feature_dict
                self.global_logits_dict[trainer.client_index]=logits_dict
                self.global_labels_dict[trainer.client_index]=labels_dict
                self.global_extracted_feature_dict_test[trainer.client_index]=extracted_feature_dict_test
                self.global_labels_dict_test[trainer.client_index]=labels_dict_test
                metrics_all['test_loss'].append(metrics[str(client_index)+' test_loss'])
                metrics_all['test_accTop1'].append(metrics[str(client_index)+' test_accTop1'])
                metrics_all['test_accTop5'].append(metrics[str(client_index)+' test_accTop5'])
            for key in metrics_all.keys():
                metrics_all[key]=float(np.mean(metrics_all[key]))
            print({"mean Test/Loss": metrics_all['test_loss']})
            print({"mean Test/AccTop1": metrics_all['test_accTop1']})
            print({"mean Test/AccTop5": metrics_all['test_accTop5']})

            self.server_trainer.client_extracted_feauture_dict=self.global_extracted_feature_dict
            self.server_trainer.client_logits_dict=self.global_logits_dict
            self.server_trainer.client_labels_dict=self.global_labels_dict
            self.server_trainer.client_extracted_feauture_dict_test=self.global_extracted_feature_dict_test
            self.server_trainer.client_labels_dict_test=self.global_labels_dict_test
            print("global training epoch ",global_epoch," start:")
            self.server_trainer.train(1)
    pass




def FedML_init():
    comm = None
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_FedGKT_distributed(process_id, worker_number, device, comm, model, train_data_local_num_dict, 
                             train_data_local_dict, test_data_local_dict, args):
    if process_id == 0:
        init_server(args, device, comm, process_id, worker_number, model)
    else:
        init_client(args, device, comm, process_id, worker_number, model, train_data_local_dict,
                    test_data_local_dict, train_data_local_num_dict)


def init_server(args, device, comm, rank, size, model):
    client_num = size - 1
    server_trainer = GKTServerTrainer(client_num, device, model, args)
    server_manager = GKTServerMananger(args, server_trainer, comm, rank, size)
    server_manager.run()
    


def init_client(args, device, comm, process_id, size, model, train_data_local_dict, test_data_local_dict,
                train_data_local_num_dict):
    client_ID = process_id - 1
    trainer = GKTClientTrainer(client_ID, train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                               device, model, args)
    client_manager = GKTClientMananger(args, trainer, comm, process_id, size)
    client_manager.run()
    