from pytools.utils_regr import *

def pre_process_data_history_WE(test_size, val_size, SEED_train, SEED_val, features, 
             feature_to_normalize, id_conv_list, target, n_turn_history, flag_single_speaker,
             flag_target_as_feat, n_batch):

    data_conv = pd.DataFrame({'id_conv':id_conv_list})
    train, validation, test = split_train_test(data_conv, test_size,val_size, SEED_train, SEED_val)
    ##PRE-PROCESSING train, val, test set 
    path_saving =  'input' + os.path.join( os.sep )

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
            dataset_dict[phase][feat] = dataset_dict[phase][feat]/ np.mean(dataset_dict[phase][feat])
# CREATE the train, validation and test set to feed the DLnet
    
    if 'DA_type' in features:
        features.extend(['back', 'sta_op', 'oth'])
        
    features = [feature for feature in features if feature != 'DA_type']
    print(features)
    turns_train = Dataset_creation_history_WE(dataset_dict['train'], dataset_dict_original['train'],
                            features, target, n_turn_history, flag_single_speaker, flag_target_as_feat)
    turns_val = Dataset_creation_history_WE(dataset_dict['validation'], dataset_dict_original['validation'], 
	                        features, target, n_turn_history, flag_single_speaker, flag_target_as_feat)
    turns_test = Dataset_creation_history_WE(dataset_dict['test'], dataset_dict_original['test'],
                            features, target, n_turn_history, flag_single_speaker, flag_target_as_feat)
# create the loader for the dataset 
    trainloader = torch.utils.data.DataLoader(
            turns_train, batch_size = n_batch, shuffle = True, collate_fn=collate )
    validationloader = torch.utils.data.DataLoader(
            turns_val, batch_size = n_batch, shuffle = True, collate_fn=collate )
    testloader = torch.utils.data.DataLoader(
            turns_test, batch_size = n_batch, shuffle = True , collate_fn=collate )
 
    return trainloader, validationloader, testloader, dataset_dict, turns_train

class Dataset_creation_history_WE(Dataset):
    def __init__(self, Dataset, Dataset_original, features, target_name,
                n_turn_history, flag_single_speaker, flag_target_as_feat):
        
        self.data = Dataset
        self.data_original = Dataset_original
        self.target_name = target_name
        self.features = features
        self.n_turn_history = n_turn_history
        self.flag_single_speaker = flag_single_speaker
        self.flag_target_as_feat = flag_target_as_feat
        tensor_InputData_dict = {}
        for idx in self.data.index:
            # select the data by index    
            sample = self.data.iloc[idx]
            features = self.features
            n_turn_history = self.n_turn_history
		
            if self.flag_target_as_feat == 0:
                feature_to_exclude = [self.target_name]
            else:
                feature_to_exclude = []
            features_context = [feature for feature in features if feature not in feature_to_exclude ]
            context, context_single_speaker, token_vectors_tot, lengths_vec_tot = create_history_sing(self.n_turn_history, features_context,
                                                                                             self.data_original, sample.label)	
            utt_embeddings = torch.nn.utils.rnn.pad_sequence(token_vectors_tot, batch_first=True)
            #prova = torch.tensor([np.array(token_vectors_tot[i]).tolist() for i in range(n_turn_history)])
            features_ = [feature for feature in features if feature not in ['label', 'utt_processed'] ]
            target_name = self.target_name
            features_c = [feature for feature in features if feature not in ['label', target_name, 'utt_processed'] ]
            #from IPython.core.debugger import Tracer; Tracer()()
            list_dic = {}
        
            if self.flag_single_speaker == 1:
                context_dict = context_single_speaker
            else:
                context_dict = context
        
            for i in range(n_turn_history):
                temp_list = list()
                if i == n_turn_history - 1:
                    features_l = features_c
                    temp_list.append(0) 
                else:
                    features_l = features_
                for feature in features_l:
                    temp_list.append(context_dict[sample.label][i][feature])
                list_dic[i] = temp_list
            tensor_InputData_dict[idx] = (torch.nn.utils.rnn.pad_sequence(token_vectors_tot, batch_first=True), torch.tensor(lengths_vec_tot), 
                                           torch.tensor([list_dic[i] for i in range(n_turn_history)]), torch.tensor(n_turn_history),
                                             torch.tensor(sample[target_name]) )
        self.tensor_InputData_dict = tensor_InputData_dict
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        # check the input is a tensor
        if type(idx) is torch.Tensor:
            idx = idx.item()
        turn_history_ = self.tensor_InputData_dict[idx]
        return turn_history_

def collate(batch):
    
    BATCH = [batch[0]]
    sequences_word__, lengths_word__, sequences__, lengths__, targets__ = zip(*BATCH)
    #from IPython.core.debugger import Tracer; Tracer()()
    
    sequences_word_ = sequences_word__[0].type(torch.DoubleTensor)
    #lengths_word_ = lengths_word__[0].type(torch.DoubleTensor)
    sequences_ = sequences__[0].type(torch.DoubleTensor)
    #lengths_ = lengths__[0].type(torch.DoubleTensor)
    targets_ = targets__[0].type(torch.DoubleTensor)     
    
    #from IPython.core.debugger import Tracer; Tracer()()
    for i in range(1, len(batch)):
        BATCH = [batch[i]]
        sequences_word, lengths_word, sequences, lengths, targets = zip(*BATCH)
        #from IPython.core.debugger import Tracer; Tracer()()
        sequences_word_ = torch.cat( (sequences_word_, sequences_word[0].type(torch.DoubleTensor)), 0)
        #lengths_word_ = torch.cat( (lengths_word_, lengths_word[0].type(torch.DoubleTensor)), 0)
        sequences_ = torch.cat( (sequences_, sequences[0].type(torch.DoubleTensor)), 0)
        #lengths_ = torch.cat( (lengths_, lengths[0].type(torch.DoubleTensor)), 0)
        targets_ = torch.cat( (targets_, targets[0].type(torch.DoubleTensor)), 0)
        
    #from IPython.core.debugger import Tracer; Tracer()()
    return sequences_word_, sequences_, targets_

def create_history_sing(n_turn_history, features, conv_AB, id_label, fix_length = 10):
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
            token_vectors = list()
            sample = conv_AB.iloc[i]
            for token in nlp(sample.utt_processed.lower()
                         #,disable= ['parser', 'tagger', 'ner'] 
                        ):
                token_vectors.append(token.vector) 
            if len(token_vectors) >= fix_length:
                #print(len(token_vectors)) 
                token_vectors = token_vectors[:fix_length]
            else:
                for i in range(fix_length - len(token_vectors)):
                    #from IPython.core.debugger import Tracer; Tracer()()
                    token_vectors.append(np.zeros(len(token.vector)))
                    
            #from IPython.core.debugger import Tracer; Tracer()()            
            lengths_vec_tot.append( torch.tensor(len(token_vectors))) 
            token_vectors_tot.append(torch.tensor(token_vectors))
##################################################
    context[id_label] = history_list
    context_single_speaker[id_label] = history_list_sing
    return context, context_single_speaker, token_vectors_tot, lengths_vec_tot

#class Model_2levels_LSTM_WE(torch.nn.Module):
#    
#    def __init__(self, input_dimensions, output_dimensions, dim_extra_feat, size_1 = 64, size_2 = 64, size_3 = 64,  layers = 1):
#        super().__init__()
#        self.seq = torch.nn.LSTM(input_dimensions, size_1, layers)
#        self.seq_1 = torch.nn.LSTM(size_1 + dim_extra_feat, size_2, layers)
#        self.layer_one = torch.nn.Linear(size_2* layers, size_3)
#        self.activation_one = torch.nn.ReLU()
#        self.layer_two = torch.nn.Linear(size_3, size_3)
#        self.activation_two = torch.nn.ReLU()
#        self.shape_outputs = torch.nn.Linear(size_3, output_dimensions)
#                    
#    def forward(self, sequences_word, lengths_word, sequences, lengths):
#        Tracer()() 
#        number_of_batches = lengths_word.shape[0]      
#        # the pack_padded_sequence is used to pad and packed the input in order to simplify the 
#        # computation. This indeed groups the zeros 
#        # inputs dimension = [sentence_length, batch_size, embedding_dim]
#        packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(
#            sequences_word, 
#            lengths_word, 
#            batch_first=False)
#        buffer, (hidden, cell) = self.seq(packed_inputs)      
#        # permute reshape the tensor changing the dimensions as specified in the parameters
#        # dimension of buffer = [batch_size, 1, hidden_dim]
#        buffer = hidden.permute(1, 0, 2)
#        # contiguous is necessary to create a unique tensor in one dimension, 
#        # while view() reshape the tensor in the dimension specified
#        #dimension of buffer = [batch_size, hidden_dim]
#        buffer = buffer.contiguous().view(number_of_batches, -1)
#        #############################################
#        
#        number_of_batches = lengths.shape[0]
#        packed = torch.nn.utils.rnn.pack_sequence(sequences) 
#        #from IPython.core.debugger import Tracer; Tracer()() 
#        buffer, (hidden, cell) = self.seq_1(packed)
#        
#        buffer = hidden.permute(1, 0, 2)
#        buffer = buffer.contiguous().view(number_of_batches, -1)
#        
#        #buffer_ = self.linear(buffer)
#        
#        buffer = self.layer_one(buffer)
#        buffer = self.activation_one(buffer)
#        buffer = self.layer_two(buffer)
#        buffer = self.activation_two(buffer)
#        buffer = self.shape_outputs(buffer)
#        return buffer
        
class Model_2levels_LSTM_WE(torch.nn.Module):
    
    def __init__(self, input_dimensions, output_dimensions, dim_extra_feat, size_1 = 64, size_2 = 64, size_3 = 64,  layers = 1):
        super().__init__()
        self.seq = torch.nn.LSTM(input_dimensions, size_1, layers)
        self.seq_1 = torch.nn.LSTM(size_1 + dim_extra_feat, size_2, layers)
        self.layer_one = torch.nn.Linear(size_2* layers, size_3)
        self.activation_one = torch.nn.ReLU()
        self.layer_two = torch.nn.Linear(size_3, size_3)
        self.activation_two = torch.nn.ReLU()
        self.shape_outputs = torch.nn.Linear(size_3, output_dimensions)
                    
    def forward(self, sequences_word, sequences, N_turn_history):
        #from IPython.core.debugger import Tracer; Tracer()()
        sequences_word = sequences_word.permute(1, 0, 2)
        buffer, (hidden, cell) = self.seq(sequences_word)
        
        ########## ADD the extra features information ###################
        sequences_word_seq = torch.cat((hidden, sequences.view(1,sequences.shape[0],sequences.shape[1])), 2)##concatenate here the packed + buffer
        sequences_word_seq = sequences_word_seq.view(int(sequences_word_seq.shape[1]/N_turn_history), N_turn_history, sequences_word_seq.shape[2])
        buffer_, (hidden, cell) = self.seq_1(sequences_word_seq)
        buffer = hidden.contiguous().view(hidden.shape[1], hidden.shape[2])
        
        buffer = self.layer_one(buffer)
        buffer = self.activation_one(buffer)
        buffer = self.layer_two(buffer)
        buffer = self.activation_two(buffer)
        buffer = self.shape_outputs(buffer)
        
        return buffer
        
#####train #######
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

            #for sequences_word, lengths_word, sequences, lengths, targets in dataloaders[phase]:    
            for sequences_word, sequences, targets in dataloaders[phase]:
                sequences_word = sequences_word.to(device)
                #lengths_word = lengths_word[0].to(device)
                sequences = sequences.to(device)
                #lengths = lengths.to(device)
                targets = targets.to(device)               
                # zero the parameter gradients
                optimizer.zero_grad()
                targets = targets.view(targets.shape[0],1)
                if phase == 'train':
                    outputs = model(sequences_word.float(), sequences.float(), targets.shape[0])
                    Tracer()()
                    loss = loss_function(outputs, targets.float())
                    loss.backward()
                    optimizer.step()
                else:
                    outputs = model(sequences, lengths, input_feat)
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

