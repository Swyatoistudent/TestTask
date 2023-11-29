import pandas as pd
import tensorflow as tf
import numpy as np



train_data = pd.read_csv('Task1/output.csv', sep=',', names=['text', 'tag'])
test_data = pd.read_csv('Task1/test_data.csv', sep=',', names=['text', 'tag'])

categories = {"O":1,"B-Mountain":2,"I-Mountain":3}
# clearing data from punctuatuion
train_data["text"] = train_data['text'].str.replace('[^\w\s]','')

train_data["tag"] = train_data["tag"].str.split()
test_data["tag"] = test_data["tag"].str.split()


train_data["tag"]=[[categories[word] for word in sentence] for sentence in train_data["tag"]]
test_data["tag"]=[[categories[word] for word in sentence] for sentence in test_data["tag"]]

train_labels = train_data["tag"]
test_labels = test_data["tag"]



from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 15000
vector_size= 300
# max conut of words
max_seq_len = 20
tokenizer = Tokenizer(oov_token="<OOV>", num_words=vocab_size)
tokenizer.fit_on_texts(train_data['text'])
sequences_train = tokenizer.texts_to_sequences(train_data['text'])
sequences_test = tokenizer.texts_to_sequences(test_data['text'])
# for lstm< vectors must have the same length
padded_train = pad_sequences(sequences_train, padding='post', maxlen=max_seq_len)
padded_test = pad_sequences(sequences_test, padding='post', maxlen=max_seq_len)
padded_labels = pad_sequences(train_labels, padding='post', maxlen=max_seq_len)
padded_labels_test = pad_sequences(test_labels, padding='post', maxlen=max_seq_len)