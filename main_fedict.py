"""
Reference:
1. https://github.com/FedML-AI
2. https://fedml.ai/
3. He, Chaoyang, et al. "Fedml: A research library and benchmark for federated machine learning." arXiv preprint arXiv:2007.13518 (2020).
@article{he2020fedml,
  title={Fedml: A research library and benchmark for federated machine learning},
  author={He, Chaoyang and Li, Songze and So, Jinhyun and Zeng, Xiao and Zhang, Mi and Wang, Hongyi and Wang, Xiaoyang and Vepakomma, Praneeth and Singh, Abhishek and Qiu, Hang and others},
  journal={arXiv preprint arXiv:2007.13518},
  year={2020}
}
4.He, Chaoyang, Murali Annavaram, and Salman Avestimehr. "Group knowledge transfer: Federated learning of large cnns at the edge." Advances in Neural Information Processing Systems 33 (2020): 14068-14080.
@article{he2020group,
  title={Group knowledge transfer: Federated learning of large cnns at the edge},
  author={He, Chaoyang and Annavaram, Murali and Avestimehr, Salman},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={14068--14080},
  year={2020}
}
"""

import argparse
import logging
import os
import socket
import sys

import numpy as np
import psutil
import setproctitle
import torch
import wandb
from collections import Counter
import torch as t
import argparse
from sys import argv
import torch

from data_loader import load_partition_data_cifar10

from resnet_client import resnet12_56,resnet4_56,resnet8_56,resnet10_56,resnet2_56
from resnet_server import resnet56_server
from FedGKTAPI import FedGKT_standalone_API

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--frequency_of_the_test', type=int, default=1)
    parser.add_argument('--backend', type=int, default=None)


    
    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=5e-4)


    parser.add_argument('--local_points', type=int, default=5000, metavar='LP',
                        help='the approximate fixed number of data points we will have on each local worker')

    parser.add_argument('--comm_round', type=int, default=300,
                        help='how many round of communications we shoud use')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--loss_scale', type=float, default=1024,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--no_bn_wd', action='store_true', help='Remove batch norm from weight decay')

    parser.add_argument('--temperature', default=3.0, type=float, help='Input the temperature: default(3.0)')
    parser.add_argument('--alpha', default=1.5, type=float, help='Input the relative weight: default(1.0)')
    parser.add_argument('--optimizer', default="SGD", type=str, help='optimizer: SGD, Adam, etc./for fedgkt etc.')
    parser.add_argument('--client_optimizer', default="sgd", type=str, help='optimizer: SGD, Adam, etc./for fedavg')

    
    
    parser.add_argument('--whether_training_on_client', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int,help='local epochs for fedavg')
    parser.add_argument('--whether_distill_on_the_server', default=1, type=int)
    parser.add_argument('--running_name', default="default", type=str)
    parser.add_argument('--sweep', default=0, type=int)
    parser.add_argument('--multi_gpu_server', action='store_true')
    parser.add_argument('--test', action='store_true',
                        help='test mode, only run 1-2 epochs to test the bug of the program')
    parser.add_argument('--client_number_per_round', default=99999,
                        help='do not change')
    parser.add_argument('--client_num_in_total', default=99999,
                        help='do not change')
    parser.add_argument('--ci', type=int, default=0,
                        help='CI')
    

    parser.add_argument('--dist_global', default=None, type=float, help='')
    parser.add_argument('--dist_locals', default=None, type=float, help='')


    parser.add_argument('--gpu_num_per_server', type=int, default=8,
                        help='gpu_num_per_server')
    parser.add_argument('--T', type=float, default=0.12,
                        help='T')
    parser.add_argument('--S', type=float, default=0.05,
                        help='S')
    parser.add_argument('--class_num', type=int, default=10,
                        help='class_num')
    parser.add_argument('--target_ent', type=int, default=3.2,
                        help='target_ent')
    parser.add_argument('--epochs_server', type=int, default=1, metavar='EP',
                        help='how many epochs will be trained on the server side')
    parser.add_argument('--epochs_client', type=int, default=1, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_number', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')
    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')#0.1 0.5 1.0 3.0
    parser.add_argument('--method', type=str, default='fedict_sim', #fedict_sim fedict_balance
                        help='method_name')
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')
    parser.add_argument('--fpkd_T', type=float, default=3.0,
                        help='fpkd_T')
    parser.add_argument('--lka_U', type=float, default=7.0,
                        help='lka_U')

    args = parser.parse_args()
    args.client_number_per_round=args.client_number
    args.client_num_in_total=args.client_number
    return args


def load_data(args, dataset_name):
    data_loader = load_partition_data_cifar10

    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, test_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num_train, class_num_test = data_loader(args.dataset, args.data_dir, args.partition_method,
                            args.partition_alpha, args.client_number, args.batch_size)

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, test_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num_train, class_num_test]
    return dataset


def create_client_model(args, n_classes,index):
    return resnet8_56(n_classes)
    #if index%5==0:
    #    client_model= resnet2_56(n_classes)
    #if index%5==1:
    #    client_model=resnet4_56(n_classes)
    #if index%5==2:
    #    client_model=resnet8_56(n_classes)
    #if index%5==3:
    #    client_model= resnet10_56(n_classes)
    #if index%5==4:
    #    client_model=resnet12_56(n_classes)
    #return client_model

def create_client_models(args, n_classes):
    client_models=[]
    for _ in range(args.client_number):
        client_models.append(create_client_model(args,n_classes,_))
    return client_models

def create_server_model(n_classes):
    #print("create server model")
    server_model = resnet56_server(n_classes)
    return server_model



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)

    # wandb.init(
    #         project="CCF_TEST2_",#"hetero-model-"+str(args.dataset)+" "+str(args.partition_alpha) if (args.client_number==10 and args.batch_size==256) else "test",#baseline#test
    #         name=str(args.dataset)+" "+str(args.partition_alpha)+" "+str(args.method),
    #         config=args,
    #         tags=""
    #     )

    seed = 1
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(5))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, test_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num_train, class_num_test] = dataset




    args.dist_global=t.zeros(args.class_num)#10类
    args.dist_locals={}
    for i in range(args.client_number):
        args.dist_locals[i]=t.zeros(args.class_num)

    print(train_data_local_num_dict)
    for i in range(args.client_number):
        y_train=[]
        y_test=[]
        for X_batch,y_batch in train_data_local_dict[i]:
            for yy in y_batch:
                y_train.append(int(yy))
        for X_batch,y_batch in test_data_local_dict[i]:
            for yy in y_batch:
                y_test.append(int(yy))

        cnt_y_train=Counter(y_train)
        cnt_y_test=Counter(y_test)
        print(i,"train",cnt_y_train)
        print(i,"test",cnt_y_test)

        for key in Counter(cnt_y_train).keys():
            args.dist_locals[i][key]+=cnt_y_train[key]
            args.dist_global[key]+=cnt_y_train[key]

    print("dist_locals:",args.dist_locals)
    print("dist_global:",args.dist_global)


    for i in range(args.client_number):
        args.dist_locals[i]=args.dist_locals[i]/float(t.sum(args.dist_locals[i]))
    args.dist_global=args.dist_global/float(t.sum(args.dist_global))

    print("dist_locals_new:",args.dist_locals)
    print("dist_global_new:",args.dist_global)

    if args.method == 'fedict_sim':
        args.method = 'lga_fd_sim'
    elif args.method == 'fedict_balance':
        args.method = 'lga_fd_balance'
    server_model=create_server_model(class_num_train)
    client_models=create_client_models(args,class_num_train)
    api=FedGKT_standalone_API(server_model,client_models,train_data_local_num_dict,test_data_local_num_dict, train_data_local_dict, test_data_local_dict, args)
    api.do_fedgkt_stand_alone(server_model,client_models,train_data_local_num_dict, test_data_local_num_dict,train_data_local_dict, test_data_local_dict, args)