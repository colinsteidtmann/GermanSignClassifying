import sys
sys.path.append("..")
import random
import gym
import math
import numpy as np
from collections import deque
from NeuralNets import ConvNet as cnn
from NeuralNets import FullyConnected as fcn
from NeuralNets import NeuralNetwork as neuralnetwork
fully_connected = fcn.FullyConnected(sizes=[1800, 800, 43], activations=["relu", "relu", "softmax"], scale_method=None, optimizer="nadam", lr=.005, lr_decay=(0.0))
fully_connected.add_dropout([1], 0.4)
convnet = cnn.ConvNet(
            conv_method="convolution",
            layer_names=["conv", "conv", "pool"],
            num_filters=[4,8,None],
            kernel_sizes=[[3,3],[3,3],None],
            stride_sizes=[[1,1],[1,1],[2,2]],
            pool_sizes=[None,None,[2,2]],
            pool_fns=[None,None,"max"],
            pad_fns=["same","same","valid"],
            activations=["relu","relu",None],
            input_channels=3,
            scale_method=None,
            optimizer="nadam",
            lr=0.001,
            lr_decay=(0.0)
        )

nn = neuralnetwork.NeuralNetwork([convnet, fully_connected], loss_fn="cross_entropy")
#print(convnet.feedforward(np.random.rand(5,30,30,3)).shape)


""" Get Data """
Cells, labels = np.load("trainCells.npy"), np.load("trainLabels.npy")
s=np.arange(Cells.shape[0])
np.random.seed(43)
np.random.shuffle(s)
Cells=Cells[s]
labels=labels[s]

len_data=len(Cells)

# Do another test/train split of the (original) train data. 
# The Test.tar-date is not used yet, but should be used
#   as the evaluation data at the end of the analysis
(X_train,X_test)=Cells[(int)(0.2*len_data):],Cells[:(int)(0.2*len_data)]
X_train = X_train.astype('float32')/255 
X_test = X_test.astype('float32')/255
train_len=len(X_train)
test_len=len(X_test)
(y_train, y_test) = labels[(int)(0.2 * len_data):], labels[:(int)(0.2 * len_data)]



""" Labels one hot encoded """
train_labels_one_hot = np.zeros((len(y_train), 43))
for idx, hot_idx in enumerate(y_train):
    train_labels_one_hot[idx, hot_idx] = 1

test_labels_one_hot = np.zeros((len(y_test), 43))
for idx, hot_idx in enumerate(y_test):
    test_labels_one_hot[idx, hot_idx] = 1

""" Training and testing """

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

n_epochs = 10
batch_size = 100
for epoch in range(n_epochs):
    for X_batch, y_batch in shuffle_batch(X_train, train_labels_one_hot, batch_size):
        nn.sgd_fit(X_batch, y_batch, shuffle_inputs=False)
    train_predictions = nn.feedforward(X_train, scale=True, test=True)
    test_predictions = nn.feedforward(X_test, scale=True, test=True)
    train_pct_correct = np.mean(np.squeeze(np.argmax(train_predictions, 1)) == np.argmax(train_labels_one_hot, 1))
    test_pct_correct = np.mean(np.squeeze(np.argmax(test_predictions, 1)) == np.argmax(test_labels_one_hot, 1))
    print(epoch, "Last batch accuracy:", train_pct_correct, "Test accuracy:", test_pct_correct)




    


