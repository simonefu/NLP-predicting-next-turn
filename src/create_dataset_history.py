from utils import *
import sys

if len(sys.argv) != 4:
	print('usage : %s <conversation> <speaker>' % sys.argv[0])
	sys.exit(1)
 
id_conv = sys.argv[1]
n_turn_history = sys.argv[2]
features = sys.argv[3]

path_saving = '..' + os.path.join(  os.sep ) + 'input' + os.path.join(  os.sep )

conv_AB.read_csv(path_saving + str(id_conv) + '.csv')

features = []
if features:
    conv_AB = conv_AB[features]
	
current_turn = {}
context = {}
context_single_speaker  = {}
#n_turn_history = 4
#features = ['lexical_density', 'mean_energy']

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
    
    