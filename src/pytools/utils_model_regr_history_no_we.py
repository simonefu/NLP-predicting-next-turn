from pytools.utils_regr import *

def create_history_sing(n_turn_history, features, conv_AB, id_label):
    current_turn = {}
    context = {}
    context_single_speaker  = {}
    
    features_ = [feature for feature in features if not feature == 'label']
    
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
        
    context[id_label] = history_list
    context_single_speaker[id_label] = history_list_sing
    return context, context_single_speaker

def create_history(n_turn_history, features , conv_AB):
    current_turn = {}
    context = {}
    context_single_speaker  = {}

    for index, row in conv_AB.iloc[0+n_turn_history: -1].iterrows():
    
        speaker_current_turn = re.search('[A-B]', row.label).group()
    
        current_turn[row.label] = {'duration' : row.end_time - row.start_time, 'word' : row.utt_processed}
        for feature in features:
            current_turn[row.label][feature] = row[feature]
        
        history_list = list()
        temporany_hist = {}
    
        history_list_sing = list()
        temporany_hist_sing = {}
        for i in range(index - n_turn_history, index):
            speaker_context_turn = re.search('[A-B]', conv_AB.iloc[i]['label']).group()
        
            temporany_hist = {feature: conv_AB.iloc[i][feature] for feature in features}
            if speaker_context_turn == speaker_current_turn:
                temporany_hist['speaker'] = 1
                temporany_hist_sing = temporany_hist
            else:
                temporany_hist['speaker'] = 0 
                temporany_hist_sing = {feature: 0 for feature in features}
                temporany_hist_sing['speaker'] = 0
            
                history_list.append(temporany_hist)
                history_list_sing.append(temporany_hist_sing)
        
        context[row.label] = history_list
        context_single_speaker[row.label] = history_list_sing
    return context, context_single_speaker

def collate(batch):
    #sort each batch by their length 
    batch.sort(key=lambda x: x[1], reverse = True)
    sequences, lengths, targets = zip(*batch)
    sequences = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True)
    targets = torch.stack(targets)
    lengths = torch.stack(lengths)
    
    return sequences, lengths, targets
	
#class Dataset_creation_history(Dataset):
class Dataset_creation_history():
    def __init__(self, Dataset, Dataset_original, features, target_name,
                n_turn_history, flag_single_speaker, flag_target_as_feat):
        
        self.data = Dataset
        self.data_original = Dataset_original
        self.target_name = target_name
        self.features = features
        self.n_turn_history = n_turn_history
        self.flag_single_speaker = flag_single_speaker
        self.flag_target_as_feat = flag_target_as_feat
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        # check the input is a tensor
        if type(idx) is torch.Tensor:
            idx = idx.item()
            
        # select the data by index    
        sample = self.data.iloc[idx]
        #from IPython.core.debugger import Tracer; Tracer()() 
        features = self.features
        n_turn_history = self.n_turn_history
		
        if self.flag_target_as_feat == 1:
           feature_to_exclude = ['utt_processed']
        else:
           feature_to_exclude = ['utt_processed', self.target_name]
        features_context = [feature for feature in features if feature not in feature_to_exclude ]		
				
        context, context_single_speaker = create_history_sing(self.n_turn_history, features_context,
                                            self.data_original, sample.label)	
        features_ = [feature for feature in features if not feature == 'label']
        target_name = self.target_name
        features_c = [feature for feature in features if feature not in ['label', target_name] ]
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
        #from IPython.core.debugger import Tracer; Tracer()()
        return (torch.tensor([list_dic[i] for i in range(n_turn_history)]), torch.tensor(n_turn_history),
                    torch.tensor(sample[target_name]) )
class LSTM_history(torch.nn.Module):
    
    def __init__(self, input_dimensions, output_dimensions, size, layers = 1):
        super().__init__()
        self.seq = torch.nn.LSTM(input_dimensions, size, num_layers = layers, bidirectional=False)
        self.linear = torch.nn.Linear(size, output_dimensions)
    
    def forward(self, inputs, lengths):
        
        number_of_batches = lengths.shape[0]
        #inputs = inputs.view(3, 4, 5)
        #packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(
        #    inputs, 
        #    lengths, 
        #    batch_first=True)
        #inputs = torch.cat(inputs).view(len(inputs), 1, -1)
        #
        packed = torch.nn.utils.rnn.pack_sequence(inputs)
        
        from IPython.core.debugger import Tracer; Tracer()() 
        buffer, (hidden, cell) = self.seq(packed)
        
        buffer = hidden.permute(1, 0, 2)
        buffer = buffer.contiguous().view(number_of_batches, -1)
        
        buffer_ = self.linear(buffer)
        
        return buffer_
		
def train_model(model, dataloaders, device, loss_function, optimizer, num_epochs=25):  
    since = time.time()

    val_loss_history = []
    train_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = 10000

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

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
                
            for sequences, lengths, acts in dataloaders[phase]:
                sequences = sequences.to(device)
                lengths = lengths.to(device)
                acts = acts.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(sequences, lengths)
                    #from IPython.core.debugger import Tracer; Tracer()()
                    outputs = outputs.view(acts.shape[0])
                    loss = loss_function(outputs, acts)
                    loss.backward()
                    optimizer.step()

                else:
                    outputs = model(sequences, lengths)
                    outputs = outputs.view(acts.shape[0])
                    #from IPython.core.debugger import Tracer; Tracer()()
                    loss = loss_function(outputs, acts)
                #_, preds = torch.max(outputs, 1)
                ## statistics
                running_loss += loss.item() * sequences.size(0)
                #running_corrects += torch.sum(preds == targets.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            #epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} '.format(phase, epoch_loss))
            #############################################
            # deep copy the model
            
            if phase == 'train':
                train_loss_history.append(epoch_loss)
            
            if phase == 'validation' and epoch_loss < best_epoch_loss:
                print(epoch_loss)
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
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
