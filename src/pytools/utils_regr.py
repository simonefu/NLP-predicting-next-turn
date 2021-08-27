import numpy as np
import pandas as pd
#import sklearn.metrics
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import os
import time
import re
import copy
#import spacy
import random 
#import neuraxle
from datetime import datetime
import json
from neuraxle.hyperparams.distributions import LogUniform
from IPython.core.debugger import Tracer

#nlp = spacy.load('..' + os.path.join(os.sep) +'open_sub_50')
#from spacy.tokenizer import Tokenizer
#nlp.tokenizer = Tokenizer(nlp.vocab)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#This function split the train and test dataset in a way that two sides of the conversations won't be in the same group
def split_train_test(Dataset, test_size,val_size, SEED_train, SEED_val):   
    train = Dataset.sample(n = round( (1 - (test_size + val_size) ) * (len(Dataset)) ), random_state = SEED_train)
    index_train = train.index
    Subset = Dataset[~Dataset.index.isin(index_train)]
    validation = Subset.sample(n = round(val_size * (len(Dataset))) , random_state = SEED_val)
    index_val = validation.index
    test = Subset[~Subset.index.isin(index_val)]
    return  train, validation, test

def create_grid_search_random(tunig_variables, tunig_var_type, n_point):
    grid_rand = {}
    for key in tunig_variables.keys():
        length_var = list()
        list_temp = list()
        compl_variable = [name for name in tunig_variables.keys() if name != key]
        for key_ in compl_variable:
            length_var.append(len(tunig_variables[key_]))
        
        for i in range(len(tunig_variables[key])-1):
            for k in range(multiplyList(length_var)*n_point):
                if tunig_var_type[key] == 'float':
                    list_temp.append(LogUniform(tunig_variables[key][i], tunig_variables[key][i+1]).rvs())
                else:
                    list_temp.append(random.randint(tunig_variables[key][i],tunig_variables[key][i+1]))
        grid_rand[key] = list_temp
    return grid_rand
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
			
def multiplyList(myList) : 
      
    # Multiply elements one by one 
    result = 1
    for x in myList: 
         result = result * x  
    return result

