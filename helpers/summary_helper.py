#----------------------------------------------------------------
# summary_helper.py does two things
# Shows the model Summary
# Prints and saves model summary in .txt file
#----------------------------------------------------------------
import os
import sys
import torch
import argparse
import torch.nn as nn
from torchinfo import summary
#from data.data import load_data



#------------------------ return num_classes--------------------

def _Summary(model: nn.Module, dataset: str, results_path: str, tag: str):

    # Aligned with the 'transformation of the original data'
    input_size = {'cifar10': (128, 3, 32, 32), 'cifar100': (1, 3, 32, 32), 'tiny-imagenet': (1, 3, 64, 64)}[dataset]
    model_name   = model.__class__.__name__


    header = f"===== {tag} | Model: {model_name} | Dataset: {dataset} =====\n"
    model_stats = summary( model,  input_size=input_size, depth=1, col_names=["input_size", "output_size", "num_params"], verbose=0 )
    summary_report = header + str(model_stats)+ "\n\n"

    print(summary_report)

    try:
        results_dir = os.path.dirname(results_path)
        os.makedirs(results_dir, exist_ok=True)
        
        with open(results_path, 'a') as f:
            f.write(summary_report)
            
        print(f"--> * Statistics successfully appended to {results_path} \n\n")

    except Exception as e:
        print(f"Error: Failed to write to file {results_path}. Reason: {e}")
