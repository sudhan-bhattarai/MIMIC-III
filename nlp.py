# %%
'''Import libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import nltk
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, GRU
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import re
pd.options.mode.chained_assignment = None

# %%
'''Load Data'''

df_a = pd.read_csv(r'C:\Users\stan\mimic-iii-data\ADMISSIONS.csv.gz')  # length of stay and id's
df_e = pd.read_csv(r'C:\Users\stan\mimic-iii-data\NOTEEVENTS.csv.gz')  # text data
df_p = pd.read_csv(r'C:\Users\stan\mimic-iii-data\PROCEDURES_ICD.csv.gz')  # procedure code for each patients
df_dp = pd.read_csv(r'C:\Users\stan\mimic-iii-data\D_ICD_PROCEDURES.csv.gz')  # procedure name
df_d = pd.read_csv(r'C:\Users\stan\mimic-iii-data\DIAGNOSES_ICD.csv.gz')  # diagnosis code for each patient
df_dd = pd.read_csv(r'C:\Users\stan\mimic-iii-data\D_ICD_DIAGNOSES.csv.gz')  # diagnosis name
df_is = pd.read_csv(r'C:\Users\stan\mimic-iii-data\ICUSTAYS.csv.gz')  # icu stay length
df_ie = pd.read_csv(r'C:\Users\stan\mimic-iii-data\INPUTEVENTS_CV.csv.gz')  # numeric entries

# %%
'''ID, Text and Length of Stay'''

df = pd.DataFrame()
df = df_a[["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME"]]
df["LENGTH_OF_STAY"] = (pd.to_datetime(df["DISCHTIME"]) - pd.to_datetime(df["ADMITTIME"])).dt.days
df = df.merge(df_e[["HADM_ID","TEXT"]], left_on = "HADM_ID", right_on = "HADM_ID")
df.drop(['DISCHTIME', 'ADMITTIME'], 1, inplace = True)
df.head()

# %%
df = df.groupby('HADM_ID').agg({'SUBJECT_ID':'first','LENGTH_OF_STAY': 'first','TEXT': ' '.join,}).reset_index()
df.head()

# %%
'''Function: Clean Text'''

replace_space = re.compile('[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~]')
clear_symbols = re.compile('[^0-9a-z #+_]')
remove_stopwords = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = replace_space.sub(' ', text)
    text = clear_symbols.sub('', text)
    text = text.replace('x', '')
    text = ' '.join(word for word in text.split() if word not in remove_stopwords)
    return text

# %%
df.TEXT = df.TEXT.apply(clean_text)
df.head()

# %%
'''Tokenize'''
max_words, max_sequence, embedding_size = 500, 250, 100
tokenizer = Tokenizer(num_words = max_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True)
X = df["TEXT"]
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X, maxlen = max_sequence)

# %%
X
# %%
'''Input and Output'''
x = X
y = df["LENGTH_OF_STAY"]

# %%
'''Train/Test Split'''
test_size = 0.2
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = test_size, random_state = 0)
print('\nNumber of training data:',x_train.shape[0])
print('\nNumber of text data:',x_test.shape[0])

#%%
'''Model structure'''

m = Sequential()
m.add(Embedding(max_words, embedding_size, input_length = 250))
m.add(LSTM(100, activation = 'relu', kernel_initializer = 'glorot_uniform', return_sequences=True))
m.add(Dropout(0.4))
m.add(LSTM(100, activation = 'relu', kernel_initializer = 'glorot_uniform', return_sequences=True))
m.add(Dropout(0.4))
m.add(LSTM(50, activation = 'tanh', dropout = 0.3, recurrent_dropout = 0.2, return_sequences = True))
m.add(GRU(70, activation = 'relu', recurrent_activation = 'tanh'))
m.add(Dense(1, activation = 'linear'))
m.summary()


# %%
'''Train the model'''

m.compile(optimizer = 'adam',loss = 'mse', metrics = ['accuracy', 'mse'])
history = m.fit(x_train, y_train, batch_size = 1000, epochs = 10, validation_split = 0.2)

# %%
