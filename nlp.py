# %%
'''Import libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import nltk
from nltk.corpus import stopwords
from keras import Model, Input, optimizers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, GRU, concatenate
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
df = df.groupby('HADM_ID').agg({'SUBJECT_ID':'first','LENGTH_OF_STAY': 'first','TEXT': ' '.join,}).reset_index()
df.head()


# %%
df2 = df
df2 = df2.merge(df_is["LOS"], left_on = "HADM_ID", right_on = df_is["HADM_ID"])
df2.head()

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
'''Clean the text data'''
df2.TEXT = df2.TEXT.apply(clean_text)
df2.head()
df2.TEXT[10]

 # %%
'''Tokenize the text data'''
max_words, max_sequence = 50000, 2000
X = df2["TEXT"]
tokenizer = Tokenizer(num_words = max_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True)
tokenizer.fit_on_texts(X.values)
X = tokenizer.texts_to_sequences(X.values)
X = pad_sequences(X, maxlen = max_sequence)
X



# %%
'''Input and Output'''
x = X                          # text input
x.shape
z = df2["LOS"].values          # numeric input
z.shape
y = df2["LENGTH_OF_STAY"]
y.shape


# %%
'''Train/Test Split'''

test_size = 0.2
x_train, x_test, z_train, z_test, y_train, y_test = train_test_split(x,z,y, test_size = test_size, random_state = 0)
# z_train, z_test = train_test_split(z, test_size = test_size, random_state = 0)
print('\nNumber of training data:',x_train.shape[0])
print('\nNumber of text data:',x_test.shape[0])


#%%
'''Model structure'''
embedding_size = 100
m = Sequential()
num_input = Input(shape = (1, ))  # numeric input layer
numl = Dense(100, activation = 'relu')(num_input)

nlp_input = Input(shape=(max_sequence,))
emb = Embedding(max_words, embedding_size, input_length = max_sequence)(nlp_input)  # text input layer
nl2 = LSTM(100, activation = 'relu', kernel_initializer = 'glorot_uniform', return_sequences=True)(emb)
d1 = Dropout(0.4)(nl2)
nl3 = LSTM(100, activation = 'relu', kernel_initializer = 'glorot_uniform', return_sequences=True)(d1)
d2 = Dropout(0.4)(nl3)
nl4 = LSTM(50, activation = 'tanh', dropout = 0.3, recurrent_dropout = 0.2, return_sequences = True)(d2)
nl5 = GRU(70, activation = 'relu', recurrent_activation = 'tanh')(nl4)

merge = concatenate([nl5, numl])   # merge Dense and LSTM layers

output = Dense(1, activation = "linear")(merge)

m = Model(inputs = [nlp_input, num_input], outputs = [output])

m.summary()


# %%
'''Train the model'''
solver = optimizers.Adam(learning_rate = 0.0001)
m.compile(optimizer = 'adam',loss = 'mse', metrics = ['accuracy'])
history = m.fit([x_train, z_train], y_train, batch_size = 1000, epochs = 1, validation_split = 0.2)


# %%
