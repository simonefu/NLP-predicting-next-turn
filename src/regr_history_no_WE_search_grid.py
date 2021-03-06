from pytools.utils_model_regr import *
from pytools.pre_process import *
from pytools.utils_model_regr_history_no_we import *
import sys

if len(sys.argv) != 4:
    print('usage : %s <conversation> <speaker>' % sys.argv[0])
    sys.exit(1)

flag_single_speaker = eval(sys.argv[1]) # True means using the speaker history turns while False means using both speaker/interlocutor history
assert isinstance(flag_single_speaker, bool), raise TypeError('param should be a bool')

n_turn_history = int(sys.argv[2])      # it is the number of previous history turns 

flag_target_as_feat = eval(sys.argv[3]) # True means using the value of the target in previous turn (using the dynamic of the target in the previous turn)
assert isinstance(flag_target_as_feat, bool), raise TypeError('param should be a bool')
    

test_size = 0.1
val_size = 0.1
SEED_train = 12
SEED_val = 1


features = ['mean_energy', 'lexical_density', 'mean_F0', 'speech_rate', 'duration', 'DA_type', 'label']
feature_to_normalize = ['mean_energy', 'mean_F0']
target = 'mean_energy'
subset_size = 0.3
SEED_subset = 10
path_saving =  'input' + os.path.join( os.sep )

if flag_target_as_feat == True:
    feat_target = 'yes'
    flag_target_feat = 1
else:
    feat_target = 'no'
if flag_single_speaker == True:
    side = 'one_side'
    flag_single_speak = 1
else:
    side = 'two_side'


id_conv_list = list()
for filename in os.listdir(path_saving):
    id_conv = re.match('\d+', filename).group(0)
    id_conv_list.append(id_conv)
print(len(id_conv_list))

data_conv_all = pd.DataFrame({'id_conv':id_conv_list})
id_conv_list_subset = data_conv_all.sample(n = round( (subset_size ) * (len(id_conv_list)) ), random_state = SEED_subset)

trainloader, validationloader, testloader, dataset_, turns_train = pre_process_data_history(test_size, val_size, SEED_train, SEED_val, features,
                                                                                       feature_to_normalize, id_conv_list_subset.id_conv.tolist(), target
                                                                                       , n_turn_history, flag_single_speak, flag_target_feat)
tunig_variables = {'lr_rate' : [0.0005, 0.005, 0.05, 0.2], 'n_layer' : [12, 52, 82,128]}
tunig_var_type = {'lr_rate' : 'float', 'n_layer' : 'int'}

loss_function = torch.nn.MSELoss()

grid_rand = create_grid_search_random(tunig_variables, tunig_var_type, n_point = 1)

size_par = 12
dataloaders = {'train' : trainloader, 'validation': validationloader, 'test' : testloader}

loss_values_val = {}
loss_values_train = {}
for i in range(size_par):
    
    learningRate = grid_rand['lr_rate'][i]
    N_size_layer = grid_rand['n_layer'][i] 
    
    model = LSTM_history(len(turns_train[6][0][0]), output_dimensions = 1, size = N_size_layer)
    model = model.to('cuda:0')  
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    model_best, val_loss_history, train_loss_history = train_model(model, dataloaders, loss_function, optimizer, num_epochs=10)
    
    loss_values_val['loss_val_' +  "{0:.4f}".format(round(learningRate,4)) + '_' + str(N_size_layer)] = val_loss_history
    loss_values_train['loss_train_' + "{0:.4f}".format(round(learningRate,4)) + '_' + str(N_size_layer)] = train_loss_history 
parameters = pd.DataFrame({'test_size': [test_size], 'val_size' : [val_size], 'SEED_train' : [SEED_train], 'SEED_val': SEED_val, 'subset_size' :subset_size, 
                          'SEED_subset': SEED_subset, 'features': [features], 'feature_to_normalize' : [feature_to_normalize], 'n_turn_history' : [n_turn_history]})

today = datetime.now()

path_saving_result = 'result'+ os.path.join( os.sep ) + 'history_no_WE' + os.path.join( os.sep ) +
        target + '_' + side + '_' + feat_target + 'as_input' + today.strftime('_%Y_%m_%d_%H_%M')

save_data(path_saving_result, loss_values_train, loss_values_val, parameters)
