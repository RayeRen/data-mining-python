
# coding: utf-8

# In[101]:

import scipy.io
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from softmax_loss import softmax_loss
from feedforward_backprop import feedforward_backprop
from utils import *
warnings.filterwarnings("ignore")


# # Load Data

# In[105]:

data = scipy.io.loadmat('data/digit_data.mat')
X = data['X']
y = data['y']
num_cases = X.shape[1]
train_num_cases = num_cases * 4 // 5
X = X.reshape([400, num_cases])
X = np.transpose(X, [1, 0])
# X has the shape of [number of samples, number of pixels]
train_data = X[:train_num_cases]
train_label = y[:, :train_num_cases]
test_data = X[train_num_cases:]
test_label = y[:, train_num_cases:]


# # Gradient Check

# In[114]:

weights = {}
weights['fully1_weight'] = np.random.randn(400, 25) / 400
weights['fully1_bias'] = np.random.randn(25, 1)
weights['fully2_weight'] = np.random.randn(25, 10) / 25
weights['fully2_bias'] = np.random.randn(10, 1)

fully1_weight_inc = np.zeros_like(weights['fully1_weight'])
fully1_bias_inc = np.zeros_like(weights['fully1_bias'])
fully2_weight_inc = np.zeros_like(weights['fully2_weight'])
fully2_bias_inc = np.zeros_like(weights['fully2_bias'])

EPSILON = 0.00010;

X = train_data[:100]
y = train_label[:, :100]
# The feedforward and backpropgation processes.
loss, _, gradients = feedforward_backprop(X, y, weights)

# check correctness of fully1_bias's gradient
for c in range(weights['fully1_bias'].shape[0]):
    weights['fully1_bias'][c, 0] = weights['fully1_bias'][c, 0] + EPSILON
    loss_2, _, gradients_2 = feedforward_backprop(X, y, weights)
    print('%.2e, %.2e, %.2e'%((loss_2 - loss) / EPSILON, gradients['fully1_bias_grad'][c, 0], gradients_2['fully1_bias_grad'][c, 0]))
    weights['fully1_bias'][c, 0]=weights['fully1_bias'][c, 0] - EPSILON


# # Train

# In[103]:

weights = {}
weights['fully1_weight'] = np.random.randn(400, 25) / 400
weights['fully1_bias'] = np.random.randn(25, 1)
weights['fully2_weight'] = np.random.randn(25, 10) / 25
weights['fully2_bias'] = np.random.randn(10, 1)

fully1_weight_inc = np.zeros_like(weights['fully1_weight'])
fully1_bias_inc = np.zeros_like(weights['fully1_bias'])
fully2_weight_inc = np.zeros_like(weights['fully2_weight'])
fully2_bias_inc = np.zeros_like(weights['fully2_bias'])

batch_size = 100
max_epoch = 10
momW = 0.9
wc = 0.0005
learning_rate = 0.1

for epoch in range(max_epoch):
    for i in range(int(np.ceil(train_num_cases / batch_size))):
        X_train = train_data[i * batch_size: (i + 1) * batch_size]
        y_train = train_label[:, i * batch_size: (i + 1) * batch_size]
        # The feedforward and backpropgation processes.
        loss, accuracy, gradients = feedforward_backprop(
            X_train, y_train, weights)
        print('%03d.%02d loss:%0.3e, accuracy:%f' % (epoch, i, loss, accuracy))

        # Updating weights
        fully1_weight_inc = get_new_weight_inc(
            fully1_weight_inc, weights['fully1_weight'], momW, wc, learning_rate, gradients['fully1_weight_grad'])
        weights['fully1_weight'] = weights['fully1_weight'] + fully1_weight_inc
        fully1_bias_inc = get_new_weight_inc(
            fully1_bias_inc, weights['fully1_bias'], momW, wc, learning_rate, gradients['fully1_bias_grad'])
        weights['fully1_bias'] = weights['fully1_bias'] + fully1_bias_inc

        fully2_weight_inc = get_new_weight_inc(
            fully2_weight_inc, weights['fully2_weight'], momW, wc, learning_rate, gradients['fully2_weight_grad'])
        weights['fully2_weight'] = weights['fully2_weight'] + fully2_weight_inc
        fully2_bias_inc = get_new_weight_inc(
            fully2_bias_inc, weights['fully2_bias'], momW, wc, learning_rate, gradients['fully2_bias_grad'])
        weights['fully2_bias'] = weights['fully2_bias'] + fully2_bias_inc


# # Test

# In[104]:

loss, accuracy, _ = feedforward_backprop(test_data, test_label, weights)
print('loss:%0.3e, accuracy:%f\n' % (loss, accuracy))

