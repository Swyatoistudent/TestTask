import pandas as pd
import tensorflow as tf
import numpy as np
from preprocesing import padded_train,padded_labels,vocab_size,vector_size,max_seq_len

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout, SpatialDropout1D,LSTM, Bidirectional
from tensorflow.keras import Sequential

def lstm_model():
    model = Sequential()
    model.add(tf.keras.layers.InputLayer((20)))
    model.add(
    Embedding(input_dim=vocab_size,
    output_dim=max_seq_len,
    input_length=max_seq_len))
    model.add(SpatialDropout1D(0.3))
    model.add(Bidirectional(LSTM(units=20, return_sequences=True,recurrent_dropout=0.2)))
    model.add(Dense(20,activation='relu'))
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model
    


callbacks = [
 keras.callbacks.EarlyStopping(monitor="val_loss",
 patience=5,
 verbose=1,
 mode="min",
 restore_best_weights=True),
 keras.callbacks.ModelCheckpoint(filepath='models/bi-lstm.ckpt',save_weights_only=True,
 verbose=1,
 save_best_only=True)
]
model = lstm_model()
model.summary()

# model.fit(padded_train,
#  padded_labels,
#  validation_split=0.2,
#  callbacks=callbacks,
#  epochs=1000)