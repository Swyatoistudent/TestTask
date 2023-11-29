from preprocesing import tokenizer,max_seq_len
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import lstm_model
import tensorflow as tf
import numpy as np

text = "'Mount Ida' in Colorado provides expansive views of the Front Range mountains."
text = text.replace('[^\w\s]','')
sequence_one = tokenizer.texts_to_sequences([text])
padded_one = pad_sequences(sequence_one,padding="post",maxlen =max_seq_len)

latest = tf.train.latest_checkpoint("models")
model = lstm_model()
model.load_weights(latest)


categor = {"O":1,"B-Mountain":2,"I-Mountain":3}
tags = {0:"miss",1:'O',2:"B-Mountain",3:"I-Mountain"}

print(text)
p = model(np.array(padded_one))
p = np.argmax(p,axis=-1)
result = [[tags[category] for category in sentence] for sentence in p[[0]]]
print(result)





# import tensorflow as tf
# import tensorflow.keras as keras
# import numpy as np
# from model import lstm_model
# from preprocesing import padded_train, train_data,test_data,padded_test,padded_labels_test
# from sklearn.metrics import mean_squared_error

# latest = tf.train.latest_checkpoint("models")
# model = lstm_model()
# model.load_weights(latest)



# y_pred_test = np.array(model.predict(padded_test))

# y_pred_test = np.argmax(y_pred_test,axis=-1)
# print("Mean_squared_error:",mean_squared_error(padded_labels_test,y_pred_test))


# print(test_data["text"][15])
# print(y_pred_test[[15]])