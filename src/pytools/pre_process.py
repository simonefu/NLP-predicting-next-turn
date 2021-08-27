from pytools.utils_model_regr import *
from pytools.utils_model_regr_history_no_we import *
import sys


def prepare_process_data(test_size, val_size, SEED_train, SEED_val, features, feature_to_normalize,
                         id_conv_list, target):

    data_conv = pd.DataFrame({'id_conv':id_conv_list})
    train, validation, test = split_train_test(data_conv, test_size,val_size, SEED_train, SEED_val)
    ##PRE-PROCESSING train, val, test set 
    path_saving =  'input' + os.path.join( os.sep )

    list_conv = {'train' : train.id_conv.tolist(), 'validation': validation.id_conv.tolist(), 'test': test.id_conv.tolist()}
    map_label_dict = {}
    dataset_dict = {}
    for phase in ['train', 'validation', 'test']:
        dict_feature_entire = {}
        for feature in features:
            dict_feature_entire[feature] = list()
        for id_conv in list_conv[phase]:
            conv_AB = pd.read_csv(path_saving + str(id_conv) + '.csv')
            conv_AB['duration'] = conv_AB['end_time'] - conv_AB['start_time']
            for feature in features:
                prev_list = dict_feature_entire[feature]
                current_list = list(conv_AB[feature])
                prev_list.extend(current_list)
                dict_feature_entire[feature] = prev_list
        map_label_dict[phase] = pd.DataFrame(dict_feature_entire['label'])
        dataset = pd.DataFrame(dict_feature_entire)
        if 'DA_type' in dataset.columns:
            type_tag = {'backchannel' : [1,0,0] , 'statement_opinion' : [0,1,0], 'other' : [0,0,1]}       
            dataset['back'] = [type_tag[i][0] for i in dataset.DA_type]
            dataset['sta_opi'] = [type_tag[i][1] for i in dataset.DA_type]
            dataset['other'] = [type_tag[i][2] for i in dataset.DA_type]
            
        #if 'DA_type' in dataset.columns:
        #    type_tag_1dim = {'backchannel' : [0] , 'statement_opinion' : [1], 'other' : [2]}       
        #    dataset['DA_type'] = [type_tag_1dim[i][0] for i in dataset.DA_type]
            
            dataset = dataset.drop(['DA_type'], axis=1)      
            dataset = dataset.dropna()      
        dataset_dict[phase] = dataset.drop(['label'], axis=1)
# NORMALIZE target values dividing for the mean value
    for phase in ['train', 'validation', 'test']:
        for feat in feature_to_normalize:
            dataset_dict[phase][feat] = dataset_dict[phase][feat]/ np.mean(dataset_dict[phase][feat])
            dataset_dict[phase][feat] = dataset_dict[phase][feat]/ np.mean(dataset_dict[phase][feat])
# CREATE the train, validation and test set to feed the DLnet
    turns_train = Dataset_creation_EW(dataset_dict['train'], target)
    turns_val = Dataset_creation_EW(dataset_dict['validation'], target)
    turns_test = Dataset_creation_EW(dataset_dict['test'], target)
# create the loader for the dataset 
    trainloader = torch.utils.data.DataLoader(
            turns_train, batch_size = 1, shuffle = True, collate_fn=collate )
    validationloader = torch.utils.data.DataLoader(
            turns_val, batch_size = len(turns_val), shuffle = True, collate_fn=collate )
    testloader = torch.utils.data.DataLoader(
            turns_test, batch_size = 32, shuffle = True, collate_fn=collate )
 
    return trainloader, validationloader, testloader, dataset_dict, turns_train
	
#def prepare_process_data_history(test_size, val_size, SEED_train, SEED_val, features,
#                                 feature_to_normalize, id_conv_list, target, flag_target_as_feat):
#
#    data_conv = pd.DataFrame({'id_conv':id_conv_list})
#    train, validation, test = split_train_test(data_conv, test_size,val_size, SEED_train, SEED_val)
#    ##PRE-PROCESSING train, val, test set 
#    path_saving =  'input' + os.path.join( os.sep )
#
#    list_conv = {'train' : train.id_conv.tolist(), 'validation': validation.id_conv.tolist(), 'test': test.id_conv.tolist()}
#    map_label_dict = {}
#    dataset_dict = {}
#    for phase in ['train', 'validation', 'test']:
#        dict_feature_entire = {}
#        for feature in features:
#            dict_feature_entire[feature] = list()
#        for id_conv in list_conv[phase]:
#            conv_AB = pd.read_csv(path_saving + str(id_conv) + '.csv')
#            conv_AB['duration'] = conv_AB['end_time'] - conv_AB['start_time']
#            for feature in features:
#                prev_list = dict_feature_entire[feature]
#                current_list = list(conv_AB[feature])
#                prev_list.extend(current_list)
#                dict_feature_entire[feature] = prev_list
#        map_label_dict[phase] = pd.DataFrame(dict_feature_entire['label'])
#        dataset = pd.DataFrame(dict_feature_entire)
#        if 'DA_type' in dataset.columns:
#            type_tag = {'backchannel' : [1,0,0] , 'statement_opinion' : [0,1,0], 'other' : [0,0,1]}       
#            dataset['back'] = [type_tag[i][0] for i in dataset.DA_type]
#            dataset['sta_opi'] = [type_tag[i][1] for i in dataset.DA_type]
#            dataset['other'] = [type_tag[i][2] for i in dataset.DA_type]
#            
#        if 'DA_type' in dataset.columns:
#            type_tag_1dim = {'backchannel' : [0] , 'statement_opinion' : [1], 'other' : [2]}       
#            dataset['DA_type'] = [type_tag_1dim[i][0] for i in dataset.DA_type]
#            
#            #dataset = dataset.drop(['DA_type'], axis=1)      
#            dataset = dataset.dropna()      
#        dataset_dict[phase] = dataset.drop(['label'], axis=1)
## NORMALIZE target values dividing for the mean value
#    for phase in ['train', 'validation', 'test']:
#        for feat in feature_to_normalize:
#            dataset_dict[phase][feat] = dataset_dict[phase][feat]/ np.mean(dataset_dict[phase][feat])
#            dataset_dict[phase][feat] = dataset_dict[phase][feat]/ np.mean(dataset_dict[phase][feat])
## CREATE the train, validation and test set to feed the DLnet
#    turns_train = Dataset_creation_history(dataset_dict['train'], target, flag_target_as_feat)
#    turns_val = Dataset_creation_history(dataset_dict['validation'], target, flag_target_as_feat)
#    turns_test = Dataset_creation_history(dataset_dict['test'], target, flag_target_as_feat)
## create the loader for the dataset 
#    trainloader = torch.utils.data.DataLoader(
#            turns_train, batch_size = 32, shuffle = True, collate_fn=collate )
#    validationloader = torch.utils.data.DataLoader(
#            turns_val, batch_size = len(turns_val), shuffle = True, collate_fn=collate )
#    testloader = torch.utils.data.DataLoader(
#            turns_test, batch_size = 32, shuffle = True, collate_fn=collate )
# 
#    return trainloader, validationloader, testloader, dataset_dict, turns_train
	
def pre_process_data_history(test_size, val_size, SEED_train, SEED_val, features, 
             feature_to_normalize, id_conv_list, target, n_turn_history, flag_single_speaker,
             			 flag_target_as_feat, n_batches):

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
            conv_AB = pd.read_csv(path_saving + str(id_conv) + '.csv')
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
        
        dataset = dataset.dropna()
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
    turns_train = Dataset_creation_history(dataset_dict['train'], dataset_dict_original['train'],
                         	features, target, n_turn_history, flag_single_speaker, flag_target_as_feat)
    turns_val = Dataset_creation_history(dataset_dict['validation'], dataset_dict_original['validation'], 
	                        features, target, n_turn_history, flag_single_speaker, flag_target_as_feat)
    turns_test = Dataset_creation_history(dataset_dict['test'], dataset_dict_original['test'],
                         	features, target, n_turn_history, flag_single_speaker, flag_target_as_feat)
# create the loader for the dataset 
    trainloader = torch.utils.data.DataLoader(
            turns_train, batch_size = n_batches, shuffle = True, collate_fn=collate )
    validationloader = torch.utils.data.DataLoader(
            turns_val, batch_size = n_batches, shuffle = True, collate_fn=collate )
    testloader = torch.utils.data.DataLoader(
            turns_test, batch_size = n_batches, shuffle = True , collate_fn=collate )
 
    return trainloader, validationloader, testloader, dataset_dict, turns_train