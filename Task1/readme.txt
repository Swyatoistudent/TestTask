The data consists of two parts: a training set and a test set. The training set is further divided into a validation set. To address this task, I employed a BI-LSTM model. For this purpose, I transformed all words into vector form using Tensorflow's Embedding.

As the input for LSTM requires vectors of equal length, I standardized them using pad_sequences() to a length of 20, as it represents the maximum length of a sentence.

The prediction is a vector of length 20, where each element (tag) corresponds to a word in the sentence.

Files descriptions:

preprocessing: Data preparation and tokenization.
model.py: Neural network architecture and training.
metrics.py: Calculation of Mean Squared Error (MSE).
test.py - testing the model on a single example

