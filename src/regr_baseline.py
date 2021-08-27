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
SEED_subset = 10
path_saving =  'input' + os.path.join( os.sep )

id_conv_list = list()
for filename in os.listdir(path_saving):
    id_conv = re.match('\d+', filename).group(0)
    id_conv_list.append(id_conv)
print(len(id_conv_list))

data_conv_all = pd.DataFrame({'id_conv':id_conv_list})
id_conv_list_subset = data_conv_all.sample(n = round( (subset_size ) * (len(id_conv_list)) ), random_state = SEED_subset)

trainloader, validationloader, testloader, dataset, turns_train = prepare_process_data(test_size, val_size, SEED_train, SEED_val, features, feature_to_normalize, 
                                                                                       id_conv_list_subset.id_conv.tolist(), target)

X_train = dataset['train']
y_train = X_train['mean_energy']

X_test = dataset['test']
y_test = X_test['mean_energy']

X_train.drop(['mean_energy', 'utt_processed'],axis=1,inplace=True)
X_test.drop(['mean_energy', 'utt_processed'],axis=1,inplace=True)

####RIDGE LINEAR REGRESSION####################
model = linear_model.Ridge()
model.fit(X_Train,y_Train)
y_predict_train = model.predict(X_Train)

mse_ridge_train = mean_squared_error(y_Train, y_predict_train)
print("Train error = "'{}'.format(mse_ridge_train)+" percent MSE Ridge Regression")

Y_test = model.predict(X_Test)
y_predict_test = list(Y_test)

mse_ridge_test = mean_squared_error(y_test, y_predict_test)
print("Test error = "'{}'.format(mse_ridge_test)+" percent MSE Ridge Regression")

####SVM ############

svm_reg = svm.SVR(kernel = 'poly', gamma = 'auto')
svm_reg.fit(X_Train, y_Train)
y1_svm = svm_reg.predict(X_Train)
y1_svm = list(y1_svm)
y2_svm = svm_reg.predict(X_Test)
y2_svm = list(y2_svm)

mse_SVM_train = mean_squared_error( y1_svm, y_Train)
print("Test error = "'{}'.format(mse_SVM_train)+" percent MSE SVM")

mse_SVM_test = mean_squared_error( y2_svm, y_Test)
print("Test error = "'{}'.format(mse_SVM_test)+" percent MSE SVM")

today = datetime.now()
path_saving_result = 'result'+ os.path.join( os.sep ) + 'no_WE_baseline_turn_' + os.path.join( os.sep ) + target +  today.strftime('_%Y_%m_%d_%H_%M')

parameters = pd.DataFrame({'test_size': [test_size], 'val_size' : [val_size], 'SEED_train' : [SEED_train], 'SEED_val': SEED_val, 'subset_size' :subset_size, 
                          'SEED_subset': SEED_subset, 'features': [features], 'feature_to_normalize' : [feature_to_normalize]})
result = pd.DataFrame({'mse_SVM_test' : [mse_SVM_test], 'mse_SVM_train': [mse_SVM_train], 'mse_ridge_train': [mse_ridge_train], 'mse_ridge_test' : [mse_ridge_test]})

parameters.to_csv(path_saving_result + os.path.join( os.sep ) + 'parameters.txt', header=True, index=False, sep='\t', mode='a')

result.to_csv(path_saving_result + os.path.join( os.sep ) + 'result.txt', header=True, index=False, sep='\t', mode='a')
