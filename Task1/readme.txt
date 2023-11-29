The data consists of two parts: a training set and a test set. The training set is further divided into a validation set. To address this task, I employed a BI-LSTM model. For this purpose, I transformed all words into vector form using Tensorflow's Embedding.

As the input for LSTM requires vectors of equal length, I standardized them using pad_sequences() to a length of 20, as it represents the maximum length of a sentence.

The prediction is a vector of length 20, where each element (tag) corresponds to a word in the sentence.

Files descriptions:

preprocessing: Data preparation and tokenization.
model.py: Neural network architecture and training.
metrics.py: Calculation of Mean Squared Error (MSE). It's a bad metric in this case, but I haven't had time to do another one.
test.py - testing the model on a single example

Link to weights: https://drive.google.com/file/d/1u_5tFLg9n8vbPWW7z9zsGsxHHOURABB3/view?usp=sharing

Possible improvements:

1) Increase the volume of training data, as the current dataset is relatively small.
2) Choose a more complex model architecture or apply a pre-trained model.
3) Implement one-hot encoding for labels.

