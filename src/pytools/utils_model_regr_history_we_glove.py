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
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import bcolz
import pickle
#import neuraxle
from datetime import datetime
import json
from neuraxle.hyperparams.distributions import LogUniform
from IPython.core.debugger import Tracer


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

def compute_vocabulary_train(path_saving, list_conv, emb_dim = 50):

    int_texts = []
    vocab = collections.defaultdict(lambda: len(vocab))
    vocab['<eos>'] = 0
    
    for id_conv in list_conv:
            conv_AB = pd.read_csv(path_saving + str(id_conv) + '.csv')    
            conv_AB = conv_AB.dropna()
            #from IPython.core.debugger import Tracer; Tracer()()
            for text in conv_AB.utt_processed:
                int_texts.append([vocab[token] for token in text.split()])

    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, emb_dim))
    words_found = 0

    for i, word in enumerate(vocab.keys()):
        try: 
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))
        vocab[word] = i
    
    print('OOV (%)' , 100 - (round(words_found/len(vocab),3) *100 ))
    
    return weights_matrix, vocab
	
def save_data(path_saving_result, loss_values_train, loss_values_val, parameters):
    
    if not os.path.exists(path_saving_result):
        os.makedirs(path_saving_result)

    with open(path_saving_result + os.path.join( os.sep ) + 'loss_val.json', 'w') as fp:
        json.dump(loss_values_val, fp)
    with open(path_saving_result + os.path.join( os.sep ) + 'loss_train.json', 'w') as fp:
        json.dump(loss_values_train, fp)    
    parameters.to_csv(path_saving_result + os.path.join( os.sep ) + 'parameters.txt', header=True, index=False, sep='\t', mode='a')

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

def pre_process_data_history_WE(test_size, val_size, SEED_train, SEED_val, features, 
             feature_to_normalize, id_conv_list, target, n_turn_history, flag_single_speaker,
             flag_target_as_feat, n_batch):

    data_conv = pd.DataFrame({'id_conv':id_conv_list})
    path_saving =  'input' + os.path.join( os.sep )
    train, validation, test = split_train_test(data_conv, test_size,val_size, SEED_train, SEED_val)
    
    weights_matrix, vocab = compute_vocabulary_train(path_saving, id_conv_list, emb_dim = 50) #compute vocabulary entire dataset
    
    ##PRE-PROCESSING train, val, test set 
    list_conv = {'train' : train.id_conv.tolist(), 'validation': validation.id_conv.tolist(), 'test': test.id_conv.tolist()}
    map_label_dict = {}
    dataset_dict = {}
    dataset_dict_original = {}
    for phase in ['train', 'validation', 'test']:
        
        dict_feature_entire = {}
        for feature in features:
            dict_feature_entire[feature] = list()

        dict_feature_entire_original = {}
        for feature in features:
            dict_feature_entire_original[feature] = list()
        
        for id_conv in list_conv[phase]:
            #if id_conv == '3056':
            #    Tracer()()
            #    print(len(conv_AB))
            conv_AB = pd.read_csv(path_saving + str(id_conv) + '.csv')    
            conv_AB = conv_AB.dropna()
            #conv_AB = conv_AB[~conv_AB.utt_processed.isin(['0'])]
            conv_AB['duration'] = conv_AB['end_time'] - conv_AB['start_time']
            conv_AB_original = conv_AB
            conv_AB = conv_AB[n_turn_history:-1]   
            for feature in features:
                prev_list = dict_feature_entire[feature]
                current_list = list(conv_AB[feature])
                prev_list.extend(current_list)
                dict_feature_entire[feature] = prev_list

                prev_list_1 = dict_feature_entire_original[feature]
                current_list_1 = list(conv_AB_original[feature])
                prev_list_1.extend(current_list_1)
                dict_feature_entire_original[feature] = prev_list_1
  
        map_label_dict[phase] = pd.DataFrame(dict_feature_entire['label'])
            
        dataset_original = pd.DataFrame(dict_feature_entire_original)
        dataset = pd.DataFrame(dict_feature_entire)
        if 'DA_type' in dataset.columns:
            type_tag = {'backchannel' : [1,0,0] , 'statement_opinion' : [0,1,0], 'other' : [0,0,1]}       
            dataset['back'] = [type_tag[i][0] for i in dataset.DA_type]
            dataset['sta_op'] = [type_tag[i][1] for i in dataset.DA_type]
            dataset['oth'] = [type_tag[i][2] for i in dataset.DA_type]
            dataset = dataset.drop(['DA_type'], axis=1)
            
        if 'DA_type' in dataset_original.columns:
            type_tag = {'backchannel' : [1,0,0] , 'statement_opinion' : [0,1,0], 'other' : [0,0,1]}       
            dataset_original['back'] = [type_tag[i][0] for i in dataset_original.DA_type]
            dataset_original['sta_op'] = [type_tag[i][1] for i in dataset_original.DA_type]
            dataset_original['oth'] = [type_tag[i][2] for i in dataset_original.DA_type]
            dataset_original = dataset_original.drop(['DA_type'], axis=1)
        
        #dataset = dataset.dropna()
        dataset_dict[phase] = dataset
        dataset_dict_original[phase] = dataset_original
# NORMALIZE target values dividing for the mean value
    for phase in ['train', 'validation', 'test']:
        dataset_dict[phase].replace(np.nan, 0, inplace=True)
        dataset_dict_original[phase].replace(np.nan, 0, inplace=True)
    
        for feat in feature_to_normalize:
            dataset_dict[phase][feat] = dataset_dict[phase][feat]/ np.mean(dataset_dict[phase][feat])
            dataset_dict_original[phase][feat] = dataset_dict_original[phase][feat]/ np.mean(dataset_dict_original[phase][feat])
# CREATE the train, validation and test set to feed the DLnet
    
    if 'DA_type' in features:
        features.extend(['back', 'sta_op', 'oth'])
        
    features = [feature for feature in features if feature != 'DA_type']
    print(features)
    turns_train = Dataset_creation_history_WE(dataset_dict['train'], dataset_dict_original['train'],
                            features, target, n_turn_history, flag_single_speaker, flag_target_as_feat, vocab)
    turns_val = Dataset_creation_history_WE(dataset_dict['validation'], dataset_dict_original['validation'], 
	                        features, target, n_turn_history, flag_single_speaker, flag_target_as_feat, vocab)
    turns_test = Dataset_creation_history_WE(dataset_dict['test'], dataset_dict_original['test'],
                            features, target, n_turn_history, flag_single_speaker, flag_target_as_feat, vocab)
# create the loader for the dataset 
    trainloader = torch.utils.data.DataLoader( turns_train, batch_size = n_batch, shuffle = True
                                                  #, collate_fn=collate 
                                                )
    validationloader = torch.utils.data.DataLoader(
                                                turns_val, batch_size = n_batch )
    testloader = torch.utils.data.DataLoader(
                                                turns_test, batch_size = n_batch)
 
    return trainloader, validationloader, testloader, dataset_dict, turns_train, weights_matrix, vocab

def create_history_sing(n_turn_history, features, conv_AB, id_label, vocab, max_len = 20):
    current_turn = {}
    context = {}
    context_single_speaker  = {}
    context_word= {}
    
    features_ = [feature for feature in features if feature not in ['label', 'utt_processed'] ]
    
    row = conv_AB[conv_AB['label'] == id_label]
    #from IPython.core.debugger import Tracer; Tracer()() 
    #speaker_current_turn = re.search('[A-B]', row.label.values[0]).group()
    speaker_current_turn = re.search('[A-B]', id_label).group()
        
    history_list = list()
    temporany_hist = {}
    
    history_list_sing = list()
    temporany_hist_sing = {}
    
    for i in range(row.index[0] - n_turn_history, row.index[0]):        
        speaker_context_turn = re.search('[A-B]', conv_AB.iloc[i]['label']).group()        
        temporany_hist = {feature: conv_AB.iloc[i][feature] for feature in features_}
        
        if speaker_context_turn == speaker_current_turn:
            temporany_hist['speaker'] = 1
            temporany_hist_sing = temporany_hist
        else:
            temporany_hist['speaker'] = 0 
            temporany_hist_sing = {feature: 0 for feature in features_}
            temporany_hist_sing['speaker'] = 0
            
        history_list.append(temporany_hist)
        history_list_sing.append(temporany_hist_sing)
##################################################
    if 'utt_processed' in features:
        token_vectors_tot = list()
        lengths_vec_tot = []
        #from IPython.core.debugger import Tracer; Tracer()()
            
        for i in range(row.index[0] - n_turn_history, row.index[0]):
            #token_vectors = list()
            int_texts = []
            
            sample = conv_AB.iloc[i]
                
            int_texts.extend([ vocab[token] for token in sample.utt_processed.split() ])
            
            if len(int_texts) >= max_len:
                int_texts = torch.LongTensor(int_texts[:max_len])
            else:
                add_pad = [0] * (max_len - len(int_texts))
                int_texts.extend( add_pad )  
                int_texts = torch.LongTensor(int_texts)
            token_vectors_tot.append(int_texts)
##################################################
    context[id_label] = history_list
    context_single_speaker[id_label] = history_list_sing
    #from IPython.core.debugger import Tracer; Tracer()() 
    return context, context_single_speaker, token_vectors_tot

class Dataset_creation_history_WE(Dataset):
    def __init__(self, Dataset, Dataset_original, features, target_name,
                n_turn_history, flag_single_speaker, flag_target_as_feat, VOCAB):
        
        self.data = Dataset
        self.data_original = Dataset_original
        self.target_name = target_name
        self.features = features
        self.n_turn_history = n_turn_history
        self.flag_single_speaker = flag_single_speaker
        self.flag_target_as_feat = flag_target_as_feat
        self.vocab = VOCAB

        tensor_InputData_dict = {}
        for idx in self.data.index:   
            sample = self.data.iloc[idx]

            if self.flag_target_as_feat == 1:
                feature_to_exclude = []
                features_ = [feature for feature in features if feature not in ['label', 'utt_processed'] ]
            else:
                feature_to_exclude = [target_name]
                features_ = [feature for feature in features if feature not in ['label', target_name, 'utt_processed'] ]
            features_.append('speaker')
                
            features_context = [feature for feature in features if feature not in feature_to_exclude ]
            ############ modify here #########################
            context, context_single_speaker, token_vectors_tot = create_history_sing(n_turn_history, features_context,
                                                                                             self.data_original, sample.label, self.vocab)	
            utt_embeddings = torch.nn.utils.rnn.pad_sequence(token_vectors_tot, batch_first=True)
            #################################################
            
            features_c = [feature for feature in features if feature not in ['label', target_name, 'utt_processed'] ]
            features_c.append('speaker')
            list_dic = {}
            if self.flag_single_speaker == 1:
                context_dict = context_single_speaker
            else:
                context_dict = context
            for i in range(n_turn_history):
                temp_list = list()
                if i == n_turn_history - 1:
                    features_l = features_c
                    if self.flag_target_as_feat == 1:
                        temp_list.append(0) 
                else:
                    features_l = features_
                    
                for feature in features_l:
                    temp_list.append(context_dict[sample.label][i][feature])
                list_dic[i] = temp_list
            #from IPython.core.debugger import Tracer; Tracer()()
            tensor_InputData_dict[idx] = (torch.nn.utils.rnn.pad_sequence(token_vectors_tot, batch_first=True), 
                                          torch.tensor([list_dic[i] for i in range(n_turn_history)]),
                                          torch.tensor([sample[target_name]]) )  
            
        self.tensor_InputData_dict = tensor_InputData_dict        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        # check the input is a tensor
        if type(idx) is torch.Tensor:
            idx = idx.item()
        turn_history_ = self.tensor_InputData_dict[idx]
        return turn_history_        

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim
    
def init_hidden(self, batch_size):
    return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

class RNN(nn.Module):
    def __init__(self, weights_matrix):
        super().__init__()
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(torch.from_numpy(weights_matrix), True)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers=1, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.decision = nn.Linear(hidden_size * 1 * 1, len(label_vocab))
        
    def forward(self, x):
        embed = self.embedding(x)
        output, hidden = self.rnn(embed)
        drop = self.dropout(hidden)
        return self.decision(drop.transpose(0, 1).contiguous().view(x.size(0), -1))

#####train #######
def train_model(model, dataloaders, loss_function, optimizer, device,  num_epochs=25):
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

            for element_tensor in dataloaders[phase]:    
                #from IPython.core.debugger import Tracer; Tracer()()
                sequences_word =  element_tensor[0]
                sequences = element_tensor[1]
                targets = element_tensor[2]
            
                sequences_word = sequences_word.to(device)
                sequences = sequences.to(device)
                targets = targets.to(device)               
                # zero the parameter gradients
                optimizer.zero_grad()
                targets = targets.view(targets.shape[0],1)
                if phase == 'train':
                    outputs = model(sequences_word, sequences)
                    loss = loss_function(outputs, targets)
                    loss.backward()
                    optimizer.step()
                else:
                    outputs = model(sequences_word, sequences)
                    #outputs = torch.tensor([i.item() for i in outputs])
                    loss = loss_function(outputs, targets)
                ## statistics
                running_loss += loss.item() * sequences.size(0)
                #running_corrects += torch.sum(preds == targets.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            
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


class Model_2levels_LSTM_WE(torch.nn.Module):
    
    def __init__(self, input_dimensions, output_dimensions, dim_extra_feat, size_3, p_drop, size_1 = 64, size_2 = 64,  layers = 1):
        super().__init__()
        
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(torch.from_numpy(weights_matrix), True)
        self.seq = torch.nn.LSTM(input_dimensions, size_1, layers, batch_first=True)
        self.drop_ = torch.nn.Dropout(p_drop)
        self.seq_1 = torch.nn.LSTM(size_1 + dim_extra_feat, size_2, layers, batch_first=True)
        self.layer_one = torch.nn.Linear(size_2* layers, size_3)
        self.activation_one = torch.nn.ReLU()
        self.layer_two = torch.nn.Linear(size_3, size_3)
        self.activation_two = torch.nn.ReLU()
        self.shape_outputs = torch.nn.Linear(size_3, output_dimensions)
                    
    def forward(self, sequences_word, sequences):
        batch_SIZE = sequences_word.shape[0]
        n_history =  sequences_word.shape[1]

        sequences_word = sequences_word.view(batch_SIZE * n_history, sequences_word.shape[2])
        embed = self.embedding(sequences_word)

        buffer, (hidden, cell) = self.seq(embed)
        
        hidden = self.drop_(hidden)
        ########## ADD the extra features information ###################
        sequences = sequences.view(1, batch_SIZE * n_history, sequences.shape[2])

        sequences_word_seq = torch.cat((hidden, sequences), 2)##concatenate here the packed + buffer
        sequences_word_seq = sequences_word_seq.view(int(sequences_word_seq.shape[1]/n_history), n_history, sequences_word_seq.shape[2])
        #from IPython.core.debugger import Tracer; Tracer()()
        buffer_, (hidden, cell) = self.seq_1(sequences_word_seq)
        buffer = hidden.contiguous().view(hidden.shape[1], hidden.shape[2])
        
        buffer = self.layer_one(buffer)
        buffer = self.activation_one(buffer)
        buffer = self.layer_two(buffer)
        buffer = self.activation_two(buffer)
        buffer = self.shape_outputs(buffer)
        
        return buffer