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
import spacy
import random 
#import neuraxle
from datetime import datetime
import json
from neuraxle.hyperparams.distributions import LogUniform
from IPython.core.debugger import Tracer

nlp = spacy.load('..' + os.path.join(os.sep) +'open_sub_50')
from spacy.tokenizer import Tokenizer
nlp.tokenizer = Tokenizer(nlp.vocab)

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

class Dataset_creation_EW_history(Dataset):
    def __init__(self, Dataset, target_name, flag_target_as_feat):
        
        self.data = Dataset
        self.target_name = target_name
        self.flag_target_as_feat = flag_target_as_feat
        
        #self.ordinals = vocab
        #self.data.TURN = self.data.utt_processed.astype(str)
        #for sample in tqdm(self.data.TURN):
            ## sample.lower take the first row of the sample not yet cycled on
        #    for token in nlp(sample.lower()
                             ##, disable = ['parser', 'tagger', 'ner']
        #                    ):
                ##create the vocabulary each time it finds a new word, add to the list ordinals
        #        if token.text not in self.ordinals:                   
        #            self.ordinals[token.text] = len(self.ordinals)       
                      
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        # check the input is a tensor
        if type(idx) is torch.Tensor:
            idx = idx.item()
        # select the data by index    
        sample = self.data.iloc[idx]
        
        target_name = self.target_name
        token_vectors = []
        for token in nlp(sample.utt_processed.lower()
                         #,disable= ['parser', 'tagger', 'ner'] 
                        ):
            token_vectors.append(token.vector)
        
        if self.flag_target_as_feat == 1:
           feature_to_exclude = ['utt_processed', target_name]
        else:
           feature_to_exclude = ['utt_processed']
        features = [feature for feature in all_features if feature not in feature_to_exclude ]		
        all_features = self.data.columns
        
        extra_features_list = sample[features]

        return (torch.tensor(token_vectors), torch.tensor(len(token_vectors)),
                    torch.tensor(sample[target_name]), torch.tensor(extra_features_list))
					
def collate(batch):
    #sort each batch by their length 
    batch.sort(key=lambda x: x[1], reverse = True)
    sequences, lengths, targets, extra_features = zip(*batch)
    sequences = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True)
    targets = torch.stack(targets)
    lengths = torch.stack(lengths)

    extra_features = torch.stack(extra_features)
    
    return sequences, lengths, targets, extra_features

class Dataset_creation_EW(Dataset):
    def __init__(self, Dataset, target_name):
        
        self.data = Dataset
        self.target_name = target_name
        
        #self.ordinals = vocab
        #self.data.TURN = self.data.utt_processed.astype(str)
        #for sample in tqdm(self.data.TURN):
            ## sample.lower take the first row of the sample not yet cycled on
        #    for token in nlp(sample.lower()
                             ##, disable = ['parser', 'tagger', 'ner']
        #                    ):
                ##create the vocabulary each time it finds a new word, add to the list ordinals
        #        if token.text not in self.ordinals:                   
        #            self.ordinals[token.text] = len(self.ordinals)       
                      
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        # check the input is a tensor
        if type(idx) is torch.Tensor:
            idx = idx.item()
        # select the data by index    
        sample = self.data.iloc[idx]
        
        target_name = self.target_name
        token_vectors = []
        for token in nlp(sample.utt_processed.lower()
                         #,disable= ['parser', 'tagger', 'ner'] 
                        ):
            token_vectors.append(token.vector)
        all_features = self.data.columns
        feature_to_exclude = ['utt_processed', target_name]
        features = [feature for feature in all_features if feature not in feature_to_exclude ]		

        extra_features_list = sample[features]

        return (torch.tensor(token_vectors), torch.tensor(len(token_vectors)),
                    torch.tensor(sample[target_name]), torch.tensor(extra_features_list))
					
def collate(batch):
    #sort each batch by their length 
    batch.sort(key=lambda x: x[1], reverse = True)
    sequences, lengths, targets, extra_features = zip(*batch)
    sequences = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True)
    targets = torch.stack(targets)
    lengths = torch.stack(lengths)

    extra_features = torch.stack(extra_features)
    
    return sequences, lengths, targets, extra_features
	
class Model_word_embedding(torch.nn.Module):
    
    def __init__(self, input_dimensions_WE, output_dimensions, input_dimension_extra_features,
                         size_1layer , layers_LSTM = 1):
        super().__init__()
        self.seq = torch.nn.LSTM(input_dimensions_WE, size_1layer, layers_LSTM)
        
        self.layer_one = torch.nn.Linear((size_1layer+input_dimension_extra_features)* layers_LSTM, size_1layer )
        self.activation_one = torch.nn.ReLU()
        self.layer_two = torch.nn.Linear(size_1layer, size_1layer)
        self.activation_two = torch.nn.ReLU()
        self.shape_outputs = torch.nn.Linear(size_1layer , output_dimensions)
    
    def forward(self, inputs, lengths, tens_features):
        
        number_of_batches = lengths.shape[0]
        
        # the pack_padded_sequence is used to pad and packed the input in order to simplify the 
        # computation. This indeed groups the zeros 
        # inputs dimension = [sentence_length, batch_size, embedding_dim]
        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(
            inputs, 
            lengths, 
            batch_first=True)
        Tracer()()
        # here it's computed the output, the hidden state and the cell of LSTM
        # dimension of the buffer = [sentence_length, batch_size, hidden_dim]
        # dimension of hidden state = [1, batch_size, hidden_dim]
        buffer, (hidden, cell) = self.seq(packed_inputs)
        #Tracer()()
        # permute reshape the tensor changing the dimensions as specified in the parameters
        # dimension of buffer = [batch_size, 1, hidden_dim]
        buffer = hidden.permute(1, 0, 2)
        #Tracer()()
        # contiguous is necessary to create a unique tensor in one dimension, 
        # while view() reshape the tensor in the dimension specified
        #dimension of buffer = [batch_size, hidden_dim]
        buffer = buffer.contiguous().view(number_of_batches, -1)
        #Tracer()()
        #############################################""
        #from IPython.core.debugger import Tracer; Tracer()()  
        
        # concatenate the buffer (output of the last LSTM) with the other features ( F0, SR, latency etc.)
        buffer_cat = torch.cat((buffer, tens_features), 1)
        
        buffer_cat = self.layer_one(buffer_cat)
        buffer_cat = self.activation_one(buffer_cat)
        buffer_cat = self.layer_two(buffer_cat)
        buffer_cat = self.activation_two(buffer_cat)
        buffer_cat = self.shape_outputs(buffer_cat)
        
        return buffer_cat

def train_model(model, dataloaders, loss_function, optimizer, num_epochs=25):
    since = time.time()

    val_loss_history = []
    train_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = 10000

    for epoch in range(num_epochs):
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
        
            # Iterate over data.
            #for sequences, lengths, acts in tqdm(dataloaders[phase]):
                
            for sequences, lengths, targets, input_feat in dataloaders[phase]:
			
                sequences = sequences.to(device)
                lengths = lengths.to(device)
                targets = targets.to(device)
                input_feat = input_feat.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                targets = targets.view(targets.shape[0],1)
                if phase == 'train':
                    outputs = model(sequences, lengths, input_feat)
                    loss = loss_function(outputs, targets)
                    loss.backward()
                    optimizer.step()

                else:
                    outputs = model(sequences, lengths, input_feat)
                    #outputs = torch.tensor([i.item() for i in outputs])
                    loss = loss_function(outputs, targets)

                #_, preds = torch.max(outputs, 1)
                ## statistics
                running_loss += loss.item() * sequences.size(0)
                #running_corrects += torch.sum(preds == targets.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            #epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            #print('{} Loss: {:.4f} '.format(phase, epoch_loss))
            #############################################
            #from IPython.core.debugger import Tracer; Tracer()() 
            # deep copy the model
            
            if phase == 'train':
                train_loss_history.append(epoch_loss)
            
            if phase == 'validation' and epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                
            if phase == 'validation':
                val_loss_history.append(epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best LOSS validation: {:4f}'.format(best_epoch_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_loss_history, train_loss_history

######################################################################
# Set Model Parameters’ .requires_grad attribute
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# This helper function sets the ``.requires_grad`` attribute of the
# parameters in the model to False when we are feature extracting. By
# default, when we load a pretrained model all of the parameters have
# ``.requires_grad=True``, which is fine if we are training from scratch
# or finetuning. However, if we are feature extracting and only want to
# compute gradients for the newly initialized layer then we want all of
# the other parameters to not require gradients. This will make more sense
# later.
# 

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
	
def save_data(path_saving_result, loss_values_train, loss_values_val, parameters):
    
    if not os.path.exists(path_saving_result):
        os.makedirs(path_saving_result)

    with open(path_saving_result + os.path.join( os.sep ) + 'loss_val.json', 'w') as fp:
        json.dump(loss_values_val, fp)
    with open(path_saving_result + os.path.join( os.sep ) + 'loss_train.json', 'w') as fp:
        json.dump(loss_values_train, fp)    
    parameters.to_csv(path_saving_result + os.path.join( os.sep ) + 'parameters.txt', header = True, index=False, sep='\t', mode='a')

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