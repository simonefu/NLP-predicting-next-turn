from pytools.utils_model_regr_history_we_glove import *
import sys

if len(sys.argv) != 4:
    print('usage : %s <conversation> <speaker>' % sys.argv[0])
    sys.exit(1)

flag_single_speaker = int(sys.argv[1])
n_turn_history = int(sys.argv[2])
flag_target_as_feat = int(sys.argv[3])

vectors = bcolz.open('../opensubtitles-en-2016.norm-swb.glove-50.dat')[:]
words = pickle.load(open('../opensub.50_words.pkl', 'rb'))
word2idx = pickle.load(open('../opensub.50_idx.pkl', 'rb'))
glove = {w: vectors[word2idx[w]] for w in words}

test_size = 0.1
val_size = 0.1
SEED_train = 12
SEED_val = 1
n_batch = 24
subset_size = 1
SEED_subset = 10
flag_single_speaker = 0
n_turn_history = 2 
flag_target_as_feat = 1
features = ['mean_energy', 'lexical_density', 'mean_F0', 'speech_rate', 'duration', 'DA_type', 'label', 'utt_processed']
feature_to_normalize = ['mean_energy', 'mean_F0']
target = 'mean_energy'
path_saving =  'input' + os.path.join( os.sep )

id_conv_list = list()
for filename in os.listdir(path_saving):
    id_conv = re.match('\d+', filename).group(0)
    id_conv_list.append(id_conv)
print(len(id_conv_list))

data_conv_all = pd.DataFrame({'id_conv':id_conv_list})
id_conv_list_subset = data_conv_all.sample(n = round( (subset_size ) * (len(id_conv_list)) ), random_state = SEED_subset)

trainloader, validationloader, testloader, dataset_, turns_train, weights_matrix, vocab = pre_process_data_history_WE(test_size, val_size, SEED_train, SEED_val, features,
                                                                                       feature_to_normalize, id_conv_list_subset.id_conv.tolist(), target
                                                                                       , n_turn_history, flag_single_speaker,flag_target_as_feat, n_batch)
dataloaders = {'train' : trainloader, 'validation': validationloader, 'test' : testloader}

tunig_variables = {'lr_rate' : [0.0005, 0.005, 0.05, 0.2], 'n_layer' : [12, 52, 82,128]}
tunig_var_type = {'lr_rate' : 'float', 'n_layer' : 'int'}

grid_rand = create_grid_search_random(tunig_variables, tunig_var_type, n_point = 1)
loss_function = torch.nn.MSELoss()
size_par = 1
p_drop = 0.2
device = torch.device('cpu') # can be changed to cuda if available
loss_values_val = {}
loss_values_train = {}
for i in range(size_par):
    
    learningRate = grid_rand['lr_rate'][i]
    N_size_layer = grid_rand['n_layer'][i] 
    
    model = Model_2levels_LSTM_WE(weights_matrix.shape[1], 1, turns_train[0][1][0].shape[0], N_size_layer, p_drop)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    model_best, val_loss_history, train_loss_history = train_model(model, dataloaders, loss_function, optimizer, device, num_epochs=7)
    loss_values_val['loss_val_' +  "{0:.4f}".format(round(learningRate,4)) + '_' + str(N_size_layer)] = val_loss_history
    loss_values_train['loss_train_' + "{0:.4f}".format(round(learningRate,4)) + '_' + str(N_size_layer)] = train_loss_history
parameters = pd.DataFrame({'test_size': [test_size], 'val_size' : [val_size], 'SEED_train' : [SEED_train], 'SEED_val': SEED_val, 'subset_size' :subset_size, 
                          'SEED_subset': SEED_subset, 'features': [features], 'feature_to_normalize' : [feature_to_normalize]})

today = datetime.now()

path_saving_result = 'result'+ os.path.join( os.sep ) + 'WE_current_turn_'  + os.path.join( os.sep ) + target +  today.strftime('_%Y_%m_%d_%H_%M')
save_data(path_saving_result, loss_values_train, loss_values_val, parameters)
torch.save(model_best.state_dict(), path_saving_result + 'model.pth')