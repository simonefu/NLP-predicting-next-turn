# NLP-predicting-next-turn

## Backgournd
The project is related to the paper "Neural representations of dialogical history for improving upcoming turn acoustic parameters prediction" presented at Interspeech in 2020. 

The aim is to predict the next turn acoustic-prosodic information during a conversation between two pairs, called the speaker and interlocutor, using previous acoustic-prosodic and/or textual information. 
The aim is comparing the use of both speaker and interlocutor vs the use of just speaker's variables. 

The target variables are:

- mean Energy
- Range Pitch
- Mean Pitch
- Speech Rate 

while input variable of previous turns are:

- Dialog acts of each turn
- Energy
- Pitch 
- Duration of the turn
- Speech Rate
- Word embeddings

The use case was implemented for the Switchboard datatset. 

## Usage example

NB. The input variables architecture must be improved. I am considering to modify it using a configuration yml file to safely and correctly pass the input arguments.

### create_dataset_history.py

It takes as input parameters the 

- ID of the conversation
- the number of turn history 
- input variables

the output is a list of the previous history for the speaker and speaker/interlocutor jointly

### regr_baseline.py

It predicts the next turn parameters using a linear regression. It serves as baseline to compare the RNN model

### regr_history_no_WE_search_grid.py

It searches for the best parameters for the RNN. The input variables are declared in the function. 

### regr_history_no_WE.py

It predicts the next turn acoustic prosodic target using a RNN architecture.

### regr_history_WE

The function predicts the next turn acoustic prosodic target using word embeddings 
