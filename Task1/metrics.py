import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from model import lstm_model
from preprocesing import padded_train, train_data,test_data

latest = tf.train.latest_checkpoint("models")
model = lstm_model()
model.load_weights(latest)

print(train_data["text"][122])
print(padded_train[122])
p = model(np.array([padded_train[122]]))
p = np.argmax(p,axis=-1)
print(p[[0]])