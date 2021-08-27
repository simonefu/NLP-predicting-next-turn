import numpy as np
import pandas as pd
import os
import time
import pickle
import copy
import spacy
import re
import sys

nlp = spacy.load('en_core_web_sm')
from spacy.tokenizer import Tokenizer
nlp.tokenizer = Tokenizer(nlp.vocab)

from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
stopWords.update(['um', 'Um', 'oh', 'uh', 'hm', 'um-hum', 'uh-huh', 'right'])

def tag_turn(name_file, path_save, TURN_DA, ID_conv_list):

    number_turns_mix = 0
    time_DA = 0
    time_turn_tot = 0
    number_tot_turn = 0
    ID = list() 
    SOURCE = list()
    ACT = list()
    TURN = list()
    START_TIME = list()
    ENERGY = list()
    
    for id_conv in ID_conv_list: 
    #for id_conv_speak in ['2008A']: 
        for ID_speaker in ['A', 'B']:
            
            id_conv_speak = id_conv + ID_speaker
            turn_da = TURN_DA[id_conv_speak]
            number_tot_turn += len(turn_da)
            #cycle on each row of the side conversation
            for row in turn_da.itertuples():
                da_type_not_null = 0
                distribution_DA = row.DA_type
        
                # compute the duration of the turn
                time_turn_tot += row.end_time - row.start_time
        
                #check if a DA type is not null, in this case compute the time od DA
                for dialog_type in distribution_DA.keys():
                    if distribution_DA[dialog_type] > 0:
                        time_DA += distribution_DA[dialog_type] * (row.end_time - row.start_time)
                        da_type_not_null += 1
                    else:
                        #print('pass')
                        pass
        
                if da_type_not_null > 1:
                    number_turns_mix += 1
        
                #find the maximum value of the DA type  
                # set TURN type
                DA_type_NTX = max(distribution_DA.items(), key=operator.itemgetter(1))[0] 
                if (DA_type_NTX == 'backchannel') |  (DA_type_NTX == 'agree') | (DA_type_NTX == 'yes'):
                    ACT.append('BAC') 
                elif (DA_type_NTX == 'statement') | (DA_type_NTX == 'opinion'):
                    ACT.append('STA') 
                #elif DA_type_NTX == 'opinion':
                #    ACT.append('OPI') 
                #elif (DA_type_NTX == 'agree') | (DA_type_NTX == 'yes'):
                #    ACT.append('AGR')             
                else:
                    ACT.append('OTH') 
                # filter and set the TURN    
        
                #m = row.utterance.rstrip()
                #m = re.sub(r'\[laughter\]', r'@', m)
                #m = re.sub(r'\[noise\]', r'#', m)
                #m = re.sub(r'\[laughter', r'[@', m)
                #m = re.sub(r'\[vocalized-noise\]', r'##', m)
                #TURN.append(m)
                TURN.append(row.utt_processed)
                
                # set the ID CONV 
                ID.append(re.sub(r'[A-B]', r'', id_conv_speak) )
        
                #set the SOURCE
                SOURCE.append(re.sub(r'(\d\d\d\d)', r'', id_conv_speak) )
                
                # energy values 
                ENERGY.append(list(row.energy_words.values()) )
                
                #add start time
                START_TIME.append(row.start_time)
                
    #ID = list(map(int, ID))  
     
    DA_turn_tag = pd.DataFrame(
            {
             'ID': ID,
             'SOURCE': SOURCE,
             'ACT': ACT, 
             'TURN': TURN,
             'ENERGY':ENERGY,
             'START_TIME': START_TIME
            })     
    #ENERGY = [(item) for item in ENERGY]
    #DA_turn_tag['ENERGY'] = ENERGY
    
    DA_turn_tag = DA_turn_tag.sort_values(by=['ID', 'START_TIME'], ascending=True)    
    #DA_turn_tag = DA_turn_tag.drop(DA_turn_tag.index[0])
        
    DA_turn_tag = DA_turn_tag[['ID', 'SOURCE', 'ACT', 'TURN', 'ENERGY']]
        
    ID = DA_turn_tag['ID'].tolist()
    ID = ['ID'] + ID
        
    SOURCE = DA_turn_tag['SOURCE'].tolist()
    SOURCE = ['SOURCE'] + SOURCE        
        
    ACT = DA_turn_tag['ACT'].tolist()
    ACT = ['ACT'] + ACT        

    TURN = DA_turn_tag['TURN'].tolist()
    TURN = ['TURN'] + TURN       
    
    #from IPython.core.debugger import Tracer; Tracer()()
    
    ENERGY = DA_turn_tag['ENERGY']
    #ENERGY = list(map(int, ENERGY))
    ENERGY = ['ENERGY'] + ENERGY
    
    with open(path_save + name_file + ".tsv","w") as f:
        for (id_conv,source,act,turn, energy) in zip(ID, SOURCE, ACT, TURN, ENERGY):
            f.write("{0}\t{1}\t{2}\t{3}\n".format(id_conv,source,act,turn, energy)) 
     
    return DA_turn_tag, time_DA, time_turn_tot, number_turns_mix, number_tot_turn
   
def tag_turn_no_label(name_file, path_save, TURN_DA, ID_conv_list):

    number_turns_mix = 0
    time_DA = 0
    time_turn_tot = 0
    number_tot_turn = 0
    ID = list() 
    SOURCE = list()
    ACT = list()
    TURN = list()
    START_TIME = list()
    ENERGY = list()
    
    for id_conv in ID_conv_list: 
    #for id_conv_speak in ['2008A']: 
        for ID_speaker in ['A', 'B']:
            
            id_conv_speak = id_conv + ID_speaker
            turn_da = TURN_DA[id_conv_speak]
            number_tot_turn += len(turn_da)
            #cycle on each row of the side conversation
            for row in turn_da.itertuples():
                # compute the duration of the turn
                time_turn_tot += row.end_time - row.start_time

                ACT.append('UNK') 

                TURN.append(row.utt_processed)
                
                # set the ID CONV 
                ID.append(re.sub(r'[A-B]', r'', id_conv_speak) )
        
                #set the SOURCE
                SOURCE.append(re.sub(r'(\d\d\d\d)', r'', id_conv_speak) )
                
                # energy values 
                ENERGY.append(list(row.energy_words.values()) )
                
                #add start time
                START_TIME.append(row.start_time)
                
    #ID = list(map(int, ID))  
     
    DA_turn_tag = pd.DataFrame(
            {
             'ID': ID,
             'SOURCE': SOURCE,
             'ACT': ACT, 
             'TURN': TURN,
             'ENERGY':ENERGY,
             'START_TIME': START_TIME
            })     
    #ENERGY = [(item) for item in ENERGY]
    #DA_turn_tag['ENERGY'] = ENERGY
    
    DA_turn_tag = DA_turn_tag.sort_values(by=['ID', 'START_TIME'], ascending=True)    
    #DA_turn_tag = DA_turn_tag.drop(DA_turn_tag.index[0])
        
    DA_turn_tag = DA_turn_tag[['ID', 'SOURCE', 'ACT', 'TURN', 'ENERGY']]
        
    ID = DA_turn_tag['ID'].tolist()
    ID = ['ID'] + ID
        
    SOURCE = DA_turn_tag['SOURCE'].tolist()
    SOURCE = ['SOURCE'] + SOURCE        
        
    ACT = DA_turn_tag['ACT'].tolist()
    ACT = ['ACT'] + ACT        

    TURN = DA_turn_tag['TURN'].tolist()
    TURN = ['TURN'] + TURN       
    
    #from IPython.core.debugger import Tracer; Tracer()()
    
    #ENERGY = ENERGY['ENERGY']
    #ENERGY = list(map(int, ENERGY))
    #ENERGY = ['ENERGY'] + ENERGY
        
    with open(path_save + name_file + ".tsv","w") as f:
        for (id_conv,source,act,turn) in zip(ID, SOURCE, ACT, TURN):
            f.write("{0}\t{1}\t{2}\t{3}\n".format(id_conv,source,act,turn)) 
     
    return DA_turn_tag, time_DA, time_turn_tot, number_turns_mix, number_tot_turn

	
import operator
def complementary(first, second):
        second = set(second)
        return [item for item in first if item not in second]

import os
import csv 
import random
import logging
# Create logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create STDERR handler
handler = logging.StreamHandler(sys.stderr)
# ch.setLevel(logging.DEBUG)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Set STDERR handler as the only handler 
logger.handlers = [handler]

# Data upload:
# input: specify the ID of the conversation, the path which contains the data,the speaker ('A' or 'B'), file type('trans' or 'word') 
# output dataframe of the transcript or the words

def download_data1(id_conv, path, speaker, tipo_file):
    if tipo_file == 'word':
        name_column = 'word'
    else: 
        name_column = 'utterance'
    doc = open(path + '\\R' + (str(id_conv)[:2]) + '\\R' + str(id_conv) + '\sw' + str(id_conv) + speaker + '-ms98-a-' + tipo_file + '.text')
    A = doc.readlines()
    df_A = pd.DataFrame(A, columns = ['raw'])
    raw = '([a-zA-Z][a-zA-Z](\d)(\d)(\d)(\d)[a-zA-Z](\-)[a-zA-Z][a-zA-Z](\d)(\d)(\-)(\w)(\-)([0-9]*))'
    Label = df_A['raw'].str.extract(raw,expand=True)
    Label.rename(columns={'':'label'}, inplace=True)
    Label.rename(columns={list(Label)[0]:'label'}, inplace=True)
    df_A['raw'] = df_A['raw'].str.replace('\s+', ' ')
    df_A['label'] = Label['label']
    df_A['start_time'] = df_A['raw'].str.extract('([0-9]*.\d\d\d\d\d\d\s[0-9]*.\d\d\d\d\d\d)', expand=True)
    df_A['start_time'] = df_A['start_time'].str.replace('(\s[0-9]*.\d\d\d\d\d\d)', '')
    df_A['end_time'] = df_A['raw'].str.extract('([0-9]*.\d\d\d\d\d\d\s[0-9]*.\d\d\d\d\d\d)', expand=True)
    df_A['end_time'] = df_A['end_time'].str.replace('([0-9]*.\d\d\d\d\d\d)\s', '')
    df_A[name_column] = df_A['raw'].str.replace('([a-zA-Z][a-zA-Z](\d)(\d)(\d)(\d)[a-zA-Z](\-)[a-zA-Z][a-zA-Z](\d)(\d)(\-)(\w)(\-)([0-9]*)(\s+)([0-9]*.\d\d\d\d\d\d(\s+)[0-9]*.\d\d\d\d\d\d)\s+)','')
    df_A['start_time'] = pd.to_numeric(df_A['start_time'])
    df_A['end_time'] = pd.to_numeric(df_A['end_time'])
    return df_A

def pre_processing_tokens(id_conv, speaker, path, label):
    
    list_token= []
    list_energy_values = []
    tipo_file = 'word'
    word_to_exclude = ['[silence] ', '[noise] ',  '[vocalized-noise] ' ]
    word_conv = download_data1(id_conv, path, speaker, tipo_file)
    word_conv = word_conv[(~word_conv['word'].isin(word_to_exclude))]
    word_conv_sel = word_conv[word_conv['label'] == label]
    
    repetition = 0 
    prev_token = '<s>'
    for token_item in word_conv_sel.itertuples():
        token = token_item.word
        
        if '[laughter]' in token:
            token = re.sub('\[laughter\]', ' ', token)
        if '-' in token:
            token = re.sub('^\[', '', token)
            token = re.sub('\[', '', token)
            token = re.sub('\]', ' ', token)
            token = re.sub('\]$', ' ', token)
            token = re.sub('laughter-', '', token)
            token = re.sub('\- $', ' ', token)
            token = re.sub('\-$', ' ', token)
            token = re.sub('^\-', ' ', token)
        if '{' in token:
            token = re.sub('{', '', token)
            token = re.sub('}', '', token)
            token = re.sub('^\[laughter-', '', token)
            token = re.sub('\]', '', token)
        if '[' in token:
            #token = re.sub('laughter', ' ', token)
            token = re.sub('\[(.*)/', ' ', token)
            token = re.sub('\[/', ' ', token)
            token = re.sub(']', ' ', token)   
        if '_1' in token:
            token = re.sub('_1', '', token)
        if '-/' in token:
            token = re.sub('(.*)\-/', ' ', token)
        if token == 'hum-um ':
            token = 'um-hum ' 
        #if token == 'i\'ll ':
        #    token = 'I\'ll ' 
        list_token.append(token)
        
        if token == prev_token:
            repetition += 1
        prev_token = token 
    utterance_processed = ''.join(word for word in list_token)
    utterance_processed = re.sub(' +', ' ', utterance_processed)
    #return [utterance_processed, list_energy_values]
    return utterance_processed, list_token, repetition
def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]
		
#######LATENCY######################################"
def compute_latency(turn_da, turn_da_interlocutor, start_pre, row):

    turn_da_interlocutor_sel = turn_da_interlocutor[(turn_da_interlocutor['start_time'] < row.start_time) & 
                                                     (turn_da_interlocutor['end_time'] > start_pre) & (row.start_time > start_pre + 0.2) &
                                                    (turn_da_interlocutor['end_time'] < row.end_time)
                                                    ]
    if not turn_da_interlocutor_sel.empty:
        latency = row.start_time - np.max(turn_da_interlocutor_sel['end_time'])
                  
    else:
        latency = 0
    start_pre = row.end_time
    return latency, start_pre
########################################################	
###ACOUSTIC COMPUTATION################################# 
def compute_acoustic_word_level(START, END, df_feature, type_feature, lim_values):
                      
    df_feature_select = df_feature[(df_feature['frameTime'] >= START) &
                                    (df_feature['frameTime'] < END)
                                    #  & (df_feature[type_feature] > 0)
                      & (df_feature[type_feature] > lim_values[0]) & (df_feature[type_feature] < lim_values[1]) 
                                              ]  
    #mean_value = np.mean(df_feature_select[type_feature]) 
    
    return df_feature_select[type_feature].tolist()
##########download, upload data####################
def download_data(n_directory,name_file):
    path = '..'+ os.path.join(os.sep)+ 'output'+ os.path.join(os.sep)+'X' + str(n_directory) + os.path.join(os.sep)
    path_name_file = str(path) + str(name_file) + '.pkl'
    file = pd.read_pickle(path_name_file)
    return file 

def upload_data(N,file,name_file):
    path = '..'+ os.path.join(os.sep)+'output'+ os.path.join(os.sep)+'X' + str(N) + os.path.join(os.sep)
    path_name_file_plk = str(path) + str(name_file) + '.pkl'
    path_name_file_csv = str(path) + str(name_file) + '.csv'    
    file.to_pickle(path_name_file_plk)
    file.to_csv(path_name_file_csv, sep='\t', encoding='utf-8')
    return path_name_file_plk 
####################################################
######Compute expected time word level #############
def linear_regression_parameter(X, test_SIZE):
# Use only one feature
    X_input = X[['lenght_utterance','distance_to_end','median_duration']]
    Y_output = X['duration']

# Split the data into training/testing sets
# Split the targets into training/testing sets

    if test_SIZE == 0:
        X_train = X_input
        Y_train = Y_output
    else:    
        X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X_input, Y_output, test_size = test_SIZE, random_state = 5)


# Create linear regression object
    regr = linear_model.LinearRegression()

# Train the model using the training sets
    regr.fit(X_train, Y_train)
    parameters = pd.DataFrame(list(zip(X_input.columns,regr.coef_)), columns= ['features','estimate_coefficients'])
    print(parameters)

    if test_SIZE == 0:
        return regr,parameters, X_train, Y_train
    else:
        
    
    
    # Make predictions using the testing set
        Y_pred = regr.predict(X_test) 

# The coefficients
# The mean squared error
        print("\nMean squared error: %.3f" % mean_squared_error(Y_test, Y_pred))
# Explained variance score: 1 is perfect prediction
        print('\nVariance score: %.2f' % r2_score(Y_test, Y_pred))

# Plot outputs
        Y_test=pd.DataFrame(Y_test)
        result = pd.concat([ Y_test, X_test], axis=1)
        index_selected = result.index.values
        df = pd.DataFrame({'index': index_selected, 'predicted_duration': Y_pred})
        df = df.set_index('index')
        result = pd.concat([ df, result], axis=1)
        result['distance_to_end'] = pd.to_numeric(result['distance_to_end'], errors='coerce').fillna(0)
        result['lenght_utterance'] = pd.to_numeric(result['lenght_utterance'], errors='coerce').fillna(0)
    
        result.plot(x = 'distance_to_end', y = 'predicted_duration', kind = 'scatter', grid=False,  alpha = 0.15,color='black', figsize= (15,10))
        result.plot(x = 'distance_to_end', y = 'duration', kind = 'scatter', grid=False,  alpha = 0.15,color='blue', figsize= (15,10))
        plt.xlabel("Distance utterance to the end (word)")
        plt.ylabel("Duration(s) ")
        plt.title("Expected & real duration vs distance utterance to the end ")
        plt.show()
    
        return regr,parameters, result, X_train, Y_train
#############################################################
###########Compute speech rate uteerance level##############
def speech_rate_computation_2(X_processed, id_utterance, word_to_exclude, word_trash):
    X_filtered_expected = X_processed[(X_processed['ID_utterance'] == id_utterance)]
    X_filtered_expected = X_filtered_expected[(~X_filtered_expected['word'].isin(word_to_exclude))]

    X_filtered_real = X_processed[(X_processed['ID_utterance'] == id_utterance)]
    X_filtered_real = X_filtered_real[(~X_filtered_real['word'].isin(word_trash))]
    
    if not (X_filtered_real.empty & X_filtered_expected.empty):
        codition_init = True
        condition_dataframe_not_empty = True
        while codition_init & condition_dataframe_not_empty:
            if X_filtered_real['word'].iloc[0] in word_to_exclude:
               indici = X_filtered_real.index
               c = indici[0]
               X_filtered_real.drop(c, inplace=True)
               condition_dataframe_not_empty = ~X_filtered_real.empty
            else:
               codition_init = False
            
        codition_init = True
        condition_dataframe_not_empty = ~X_filtered_real.empty
        while codition_init & condition_dataframe_not_empty:
            if X_filtered_real['word'].iloc[-1] in word_to_exclude:
               indici = X_filtered_real.index
               c = indici[-1]
               X_filtered_real.drop(c, inplace=True)
               condition_dataframe_not_empty = ~X_filtered_real.empty
            else:
               codition_init = False   
       
        real_duration = X_filtered_real.groupby(['ID_utterance'])['duration'].sum()
        expected_duration = X_filtered_expected.groupby(['ID_utterance'])['expected_duration'].sum()
        pointwise_speech_rate = real_duration/expected_duration
        pointwise_speech_rate = pointwise_speech_rate.item()
    else:
   	    pointwise_speech_rate = 0  
        
    return pointwise_speech_rate 
	
#########COMPUTE LEXICAL DENSITY TURN LEVEL############
def compute_lexical_density(list_token, stopWords):
    counts = {}
    for word in list_token:
        word = re.sub(' +', '', word)
        if (word not in counts) & (word not in stopWords):
            counts[word] = 1
        elif word not in stopWords:
            counts[word] += 1
    lexical_density = np.sum(list(counts.values()))/len(list_token)
    return lexical_density
########COMPUTE ONE SIDE CONVERSATION#########################
def create_one_side_conv(id_conv, id_speaker): 
    
    path_trans = '..' + os.path.join(  os.sep ) + str('..') + os.path.join( os.sep , 'swb_ms98_transcriptions')
    #path = '..\swb_ms98_transcriptions'
    path_DA = 'conversation_DA_label' + os.path.join(  os.sep )
    path_SR =  'conversation_speech_rate' + os.path.join(  os.sep )
    tipo_file = 'trans'
    path_saving = 'input' + os.path.join(  os.sep )
    
    word_to_exclude = ['[silence]', '[noise]', '[laughter]', 'um', '[vocalized-noise]', 'Um','','oh', 'uh']
    word_trash = ['[noise]', '[laughter]', '[vocalized-noise]' ]

    #############Dialog acts######################################
    turn_da_sing = pd.read_csv(path_DA + str(id_conv) + str(id_speaker) +'.csv')
    #############LANTENCY##########################################
    turn_da = download_data1(str(id_conv), path_trans, id_speaker, tipo_file)
    turn_da = turn_da[['label', 'start_time', 'end_time', 'utterance']]
    turn_da = turn_da[~turn_da['utterance'].isin(['[noise] ', '[laughter] ', '[silence] '])]
            
    turn_da_interlocutor = download_data1(str(id_conv), path_trans, diff(['A', 'B'], [id_speaker])[0], tipo_file)
    turn_da_interlocutor = turn_da_interlocutor[['label', 'start_time', 'end_time', 'utterance']]
    turn_da_interlocutor = turn_da_interlocutor[~turn_da_interlocutor['utterance'].isin(['[noise] ', '[laughter] ', '[silence] '])]
    
    ############SPEECH RATE######################################
    #X_sing_conv = X_processed_new[(X_processed_new['ID_conversation'] == str(id_conv)) & (X_processed_new['ID_speaker'] == id_speaker)]
    X_sing_conv = pd.read_csv(path_SR + str(id_conv) + str(id_speaker) +'.csv')   
    ########################################################
    ####ENERGY##############################################
    type_file = 'energy'
    lim_values = [0, 3]
    type_feature = 'pcm_RMSenergy'

    treshold_p_voicing = 0.6
    type_file_filter = 'prosody'
    filter_feature= 2

    df_feature = pd.read_csv('..' +os.path.join(  os.sep ) + '..' +os.path.join(  os.sep ) + 'opensmile-50ms' + os.path.join(  os.sep ) + 'sw0' + str(id_conv) + '_' +
                                                 str(id_speaker) + '.' + str(type_file) +'.csv', sep = ';')
                        
    if filter_feature == 2:
        [av_tot, median_tot, df_feature_filter] = normilize_all_conv(id_conv, id_speaker,treshold_p_voicing,
                                                                 type_file_filter,
                                                                 type_file, type_feature)
        df_feature_filter[type_feature] = df_feature_filter[type_feature]/av_tot
                
        DF_FEATURE_energy = df_feature_filter    
    elif filter_feature == 1:
        [av_tot, median_tot, df_feature_filter] = normilize_all_conv(id_conv,
                                                                id_speaker,treshold_p_voicing,
                                                                 type_file_filter,
                                                                 type_file, type_feature)
        DF_FEATURE_energy = df_feature_filter                  
                            
    elif filter_feature == 0:
        DF_FEATURE_energy = df_feature
    else:
        print('the flag must be  2, 1, 0 (read the descprition)')            
            
          
    ########################################################
    ####PROSODY##############################################
    type_file = 'prosody'
    lim_values = [80, 350]
    type_feature = 'F0final_sma'

    treshold_p_voicing = 0.6
    type_file_filter = 'prosody'
    filter_feature= 2

    df_feature = pd.read_csv('..' +os.path.join(  os.sep ) + '..' +os.path.join(  os.sep )+ 'opensmile-50ms' + os.path.join(  os.sep ) + 'sw0' + str(id_conv) + '_' +
                                                 str(id_speaker) + '.' + str(type_file) +'.csv', sep = ';')
                        
    if filter_feature == 2:
        [av_tot, median_tot, df_feature_filter] = normilize_all_conv(id_conv, id_speaker,treshold_p_voicing,
                                                                 type_file_filter,
                                                                 type_file, type_feature)
        df_feature_filter[type_feature] = df_feature_filter[type_feature]/av_tot
                
        DF_FEATURE_prosody = df_feature_filter    
    elif filter_feature == 1:
        [av_tot, median_tot, df_feature_filter] = normilize_all_conv(id_conv,
                                                                id_speaker,treshold_p_voicing,
                                                                 type_file_filter,
                                                                 type_file, type_feature)
        DF_FEATURE_prosody = df_feature_filter                  
                            
    elif filter_feature == 0:
        DF_FEATURE_prosody = df_feature
    else:
        print('the flag must be  2, 1, 0 (read the descprition)')            
            
          
    ########################################################
    start_pre = 0
    for row in turn_da.itertuples():
        #print("[ %s , %s]" % (row.start_time, row.end_time))
        start_turn = row.start_time
        end_turn = row.end_time
                
        utt_processed, list_token, repetition = pre_processing_tokens(id_conv, id_speaker, path_trans, row.label)
        
        lexical_density = compute_lexical_density(list_token, stopWords)
                
        energy_turn = compute_acoustic_word_level(row.start_time, row.end_time, DF_FEATURE_energy,
                                    'pcm_RMSenergy',
                                    [0, 3])

                #list_energy_values.append(word_energy)
                
        prosody_turn = compute_acoustic_word_level(row.start_time, row.end_time, DF_FEATURE_prosody,
                                    'F0final_sma',
                                    [80, 350])

        latency, start_pre = compute_latency(turn_da, turn_da_interlocutor, start_pre, row)
        
        id_utterance = re.findall('^.*-([0-9]+)$',row.label)[0]
        id_utterance = re.sub('^[0]+', '', id_utterance)
        speech_rate = speech_rate_computation_2(X_sing_conv, int(id_utterance), word_to_exclude, word_trash)
        
        try:
              turn_da.loc[[row.Index], 'utt_processed' ] = utt_processed
        except:
              turn_da.loc[[row.Index], 'utt_processed' ] = np.nan
        
        try:
              turn_da.loc[[row.Index], 'number_of_words' ] = len(list_token)
        except:
              turn_da.loc[[row.Index], 'number_of_words' ] =  np.nan

        try:
              turn_da.loc[[row.Index], 'repetition' ] = int(repetition)
        except:
              turn_da.loc[[row.Index], 'repetition' ] = np.nan
        try:
              turn_da.loc[[row.Index], 'lexical_density' ] = lexical_density
        except:
              turn_da.loc[[row.Index], 'lexical_density' ] = np.nan
        try:
              turn_da.loc[[row.Index], 'mean_energy' ] = np.mean(energy_turn)
        except:
              turn_da.loc[[row.Index], 'mean_energy' ] = np.nan
        try:
              turn_da.loc[[row.Index], '25perc_energy' ] = np.percentile(energy_turn, 25)
        except:
              turn_da.loc[[row.Index], '25perc_energy' ] = np.nan
        try:
              turn_da.loc[[row.Index], '75perc_energy' ] = np.percentile(energy_turn, 75)
        except:
              turn_da.loc[[row.Index], '75perc_energy' ] =  np.nan
        try:
              turn_da.loc[[row.Index], 'mean_F0' ] = np.mean(prosody_turn)
        except:
              turn_da.loc[[row.Index], 'mean_F0' ] = np.nan
        try:
              turn_da.loc[[row.Index], '25perc_F0' ] = np.percentile(prosody_turn, 25)
        except:
              turn_da.loc[[row.Index], '25perc_F0' ] = np.nan
        try:
              turn_da.loc[[row.Index], '75perc_F0' ] = np.percentile(prosody_turn, 75)
        except:
              turn_da.loc[[row.Index], '75perc_F0' ] = np.nan
        try:
              turn_da.loc[[row.Index], 'latency' ] = latency
        except:
              turn_da.loc[[row.Index], 'latency' ] = np.nan
        try:
              turn_da.loc[[row.Index], 'speech_rate' ] = speech_rate
        except:
              turn_da.loc[[row.Index], 'speech_rate' ] = np.nan
        try:
              turn_da.loc[[row.Index], 'DA_type' ] = turn_da_sing[turn_da_sing['label'] == row.label].niteType.item()
        except:
              turn_da.loc[[row.Index], 'DA_type' ] = np.nan
        
    #turn_da.to_csv(path_saving + str(id_conv) +'_' + id_speaker + '.csv')
    return turn_da        
 
def normilize_all_conv(id_conv, id_speaker,treshold_p_voicing, type_file_filter, type_file_variable, variable_name):

    df_feature_filter =  pd.read_csv('..' +os.path.join(  os.sep ) + 'opensmile-50ms' + os.path.join(  os.sep ) + 'sw0' + str(id_conv) + '_' +
                                                 str(id_speaker) + '.' + str(type_file_filter) +'.csv', sep = ';')
    # tag rows based on the threshold
    df_feature_filter['tag'] = df_feature_filter['voicingFinalUnclipped_sma'] > treshold_p_voicing

    
    # first row is a True preceded by a False
    fst = df_feature_filter.index[df_feature_filter['tag'] & ~df_feature_filter['tag'].shift(1).fillna(False)]

    # last row is a True followed by a False
    lst = df_feature_filter.index[df_feature_filter['tag']& ~df_feature_filter['tag'].shift(-1).fillna(False)]

    # filter those which are adequately apart
    pr = [(i, j) for i, j in zip(fst, lst) if j > i + 2]


    df_variable = pd.read_csv('..' +os.path.join(  os.sep ) + 'opensmile-50ms' + os.path.join(  os.sep ) + 'sw0' + str(id_conv) + '_' +
                                                 str(id_speaker) + '.' + str(type_file_variable) +'.csv', sep = ';')
    df_feature = pd.read_csv('..' +os.path.join(  os.sep ) + 'opensmile-50ms' + os.path.join(  os.sep ) + 'sw0' + str(id_conv) + '_' +
                                                 str(id_speaker) + '.' + str(type_file_filter) +'.csv', sep = ';')    
    
    values_voiced = list()
    index_filtered_df_variable = list()
    for diad_index in pr:

        t_start = df_feature.loc[diad_index[0]].frameTime.item()
        t_end = df_feature.loc[diad_index[1]].frameTime.item()
        
        df_selected_variable = df_variable[(df_variable['frameTime'] > t_start)  
             &  (df_variable['frameTime'] < t_end)]
        
        values_voiced.extend( (df_selected_variable[variable_name]).tolist() )
        
        index_filtered_df_variable.extend( df_variable.index[(df_variable['frameTime'] > t_start)  
             &  (df_variable['frameTime'] < t_end)] )
        
    #from IPython.core.debugger import Tracer; Tracer()()    
    average_tot = np.mean(values_voiced)
    median_tot = np.median(values_voiced) 
    df_variable = df_variable.loc[index_filtered_df_variable]
    
    return [average_tot, median_tot, df_variable] 
 