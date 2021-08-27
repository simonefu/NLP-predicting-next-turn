from pytools.utils_model_regr import *
from pytools.pre_process import *
import sys

if len(sys.argv) != 1:
	print('usage : %s <conversation> <speaker>' % sys.argv[0])
	sys.exit(1)
test_size = 0.1
val_size = 0.1
SEED_train = 12
SEED_val = 1
features = ['mean_energy', 'lexical_density', 'mean_F0', 'speech_rate', 'duration', 'DA_type', 'label', 'utt_processed']
feature_to_normalize = ['mean_energy', 'mean_F0']
target = 'mean_energy'
subset_size = 1
SEED_subset = 20
path_saving =  'input' + os.path.join( os.sep )
id_conv_list = list()
for filename in os.listdir(path_saving):
    id_conv = re.match('\d+', filename).group(0)
    id_conv_list.append(id_conv)
print(len(id_conv_list))
data_conv_all = pd.DataFrame({'id_conv':id_conv_list})
id_conv_list_subset = data_conv_all.sample(n = round( (subset_size ) * (len(id_conv_list)) ), random_state = SEED_subset)
trainloader, validationloader, testloader, dataset, turns_train = prepare_process_data(test_size, val_size, SEED_train, SEED_val, features, feature_to_normalize, id_conv_list_subset.id_conv.tolist(), target)
dataloaders = {'train' : trainloader, 'validation': validationloader, 'test' : testloader}
loss_function = torch.nn.MSELoss()
learningRate = 0.001
N_size_layer = 42 
model = Model_word_embedding(turns_train[0][0].shape[1], 1, turns_train[0][3].shape[0], size_1layer = N_size_layer)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = model.cuda() 
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
model_best, val_loss_history, train_loss_history = train_model(model, dataloaders,
                                                               loss_function, optimizer, num_epochs=15)
parameters = pd.DataFrame({'test_size': [test_size], 'val_size' : [val_size],
                           'SEED_train' : [SEED_train], 'SEED_val': SEED_val,
						   'subset_size' :subset_size, 
                          'SEED_subset': SEED_subset, 'features': [features], 
                            'feature_to_normalize' : [feature_to_normalize]})
today = datetime.now()
path_saving_result = 'result' + os.path.join( os.sep ) + 'WE_current_turn_' + target + today.strftime('_%Y_%m_%d_%H_%M')
save_data(path_saving_result, train_loss_history, val_loss_history, parameters)
torch.save(model_best.state_dict(), path_saving_result + 'model_best.pth')

running_loss= 0.0
with torch.no_grad():
    model_best.eval()
    for sequences, lengths, targets, input_feat in dataloaders['test']:
	    sequences = sequences.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device)
		input_feat = input_feat.to(device)
        results = model_best(sequences, lengths, input_feat)
		results = results.view(targets.shape[0]) # change dimension 
        loss = loss_function(results, targets)
        running_loss += loss.item() * (sequences.size(0))
epoch_loss = running_loss / len(dataloaders['test'].dataset)
print(epoch_loss)