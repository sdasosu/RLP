#====== This is the main controller file of this project =========-

# python controller.py --model resnet18  --dataset cifar100 --pruning Flase
#
#----------------------------------------------------------------
# data --> summary --> model --> run --> save 
# load --> summary --> prune --> run --> save
#----------------------------------------------------------------
#--------------------- Suppring models --------------------------
# VGG FAMILY:        vgg11, vgg11_bn , vgg16 , vgg19 , vgg19_bn
# RESNET FAMILY:     resnet18, resnet34, resnet50, resnet101, resnet152
# MOBILENET FAMILY:  mobilenet_v3_small, mobilenet_v3_large
# DENSENET FAMILY:   densenet121 , densenet161 , densenet169 , densenet201 , 
# GOOGLENET FAMILY:  googlenet [NEED FIX]
# RESNEXT FAMILY :   resnext50_32x4d , resnext101_32x8d 
#----------------------------------------------------------------


#--------------------- Supporting Data sets ---------------------
# cifar10, cifar100,tiny-imagenet
#----------------------------------------------------------------


#--------------------- identifications of files -----------------
#
#
#


#================================================================
import os
import sys
import torch
import argparse
import torch.nn as nn
from pprint import pprint
# from helpers import graph_maker
from helpers import model_zoo
from helpers.data_factory import get_data
from helpers.summary_helper import _Summary
from helpers.train_test_helper import train_test
from pruning_utils.mod_to_table import build_pruning_table , print_table


#----------------------------------------------------------------
# Keep the controller super clean 
#================================================================


#========================== Extras ==============================
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint_dir    = './assets/checkpoint/baselines'
results_dir       = './assets/results'
table_dir         = './assets/table'

os.makedirs(table_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)


#================================================================



#================================================================
parser = argparse.ArgumentParser(description='Simple Model Training and Testing')
parser.add_argument('--model',   type=str, required=True, help='Model name to train')
parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'cifar100', 'tiny-imagenet'], help='Dataset to use')
parser.add_argument('--epochs',  type=int, required=False, default=1,  help='Epochs to train')
args = parser.parse_args()
#================================================================



#===================== Overall Process ==========================
print(f"==== Loading model {args.model} , Data {args.dataset} ======")

#-----------------------------------------------------------------------------------------
model_function = getattr(model_zoo, args.model) 
num_classes = {'cifar10': 10, 'cifar100': 100, 'tiny-imagenet': 200}[args.dataset]
model = model_function(in_channels=3, num_classes=num_classes)
#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
checkpoint_path = os.path.join(checkpoint_dir, f'{args.model}_{args.dataset}_best.pt')      #
results_path    = os.path.join(results_dir, f'{args.model}_{args.dataset}.txt')             #
table_dir       = os.path.join(table_dir, f'{args.model}_{args.dataset}_table.csv')         #
#-----------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------
_Summary(model=model,  dataset=args.dataset, results_path=results_path, tag="Full Size Model")
train_test(model, args.dataset, checkpoint_path, results_path, args.epochs)
#-----------------------------------------------------------------------------------------




#------------- Model to -> table , json and pt(weights)
table = build_pruning_table(model, table_path=table_dir, default_ratio=0.1)
print_table(table)
#========================================================================================

