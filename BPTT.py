import numpy as np
from numpy.core.tests.test_mem_overlap import xrange
import tensorflow as tf


def range_matrix(r,c):
    return np.arange(r*c).reshape((r, c))*0.1+0.1


input_len = 3
num_classes = 3
n, p = 0, 0
hidden_size = 2 # size of hidden layer of neurons
seq_length = 3 # number of steps to unroll the RNN for
learning_rate = 1

data_len = 50000
x = np.arange(data_len)+1

ground_truth = [(x[i-1] + x[i-2]) % 3 for i in range(data_len)]

# model parameters
U = range_matrix(hidden_size, input_len) # input to hidden
W = range_matrix(hidden_size, hidden_size) # hidden to hidden
V = range_matrix(num_classes, hidden_size) # hidden to output
bs = np.zeros((hidden_size, 1)) # hidden bias
bo = np.zeros((num_classes, 1)) # output bias

# 完成下面的函数，将代码填写到 '''Fill your code HERE'''的地方

def forward_and_backprop(inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    # forward pass
    for t in xrange(seq_length):
        xs[t] = inputs[t:t + 3].reshape(input_len, 1)  # make a matrix(rank 2)
        hs[t] = np.tanh(np.dot(U, xs[t]) + np.dot(W, hs[t-1]) + bs)  # 计算hidden state。激活函数使用tanh
        ys[t] = np.dot(V, hs[t]) + bo  # 计算output logits。注意这里没有激活函数，我们将在下一步计算softmax
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # softmax
        loss = -tf.reduce_sum(targets * ps[t])  # 计算交叉熵

    # print("@@@@@@@@xs@@@@@@")
    # print(xs)
    #
    # print("@@@@@@@@hs@@@@@@")
    # print(hs)
    #
    # print("@@@@@@@@ys@@@@@@")
    # print(ys)
    #
    # print("@@@@@@@@ps@@@@@@")
    # print(ps)
    #
    # print("@@@@@@@@loss@@@@@@")
    # print(loss)

    # 反向传播过程
    dU, dW, dV = np.zeros_like(U), np.zeros_like(W), np.zeros_like(V)
    dbs, dbo = np.zeros_like(bs), np.zeros_like(bo)
    dhnext = np.zeros_like(hs[0])

    for t in reversed(xrange(seq_length)):
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print("$$$$$$$$$$$$$$$$$第%d次 BP" % t)
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        dy = np.copy(ps[t])
        # print("@@@@@@@@ dy @@@@@@")
        # print(dy)

        dy[targets[t]] -= 1  # softmax-交叉熵delta： y-t
        # print("@@@@@@@@ dy @@@@@@")
        # print(dy)

        # print("@@@@@@@@ s3.T @@@@@@" % np.transpose(hs[t]))
        # print(np.transpose(hs[t]))

        dV += np.dot(dy, np.transpose(hs[t]))  # V-nabla
        print("@@@@@@@@ dV-%d @@@@@@" % t)
        print(dV)

        dbo += dy  # bo-nabla
        # print("@@@@@@@@ dbo @@@@@@")
        # print(dbo)

        if t == 2:
            dh = np.dot(np.transpose(V), dy)  # backprop into hidden-state
            # print("@@@@@@@@ dh @@@@@@")
            # print(dh)
        else:
            dh = np.dot(np.transpose(W), dhnext) + np.dot(np.transpose(V), dy)  # backprop into hidden-state
            # print("@@@@@@@@ dh @@@@@@")
            # print(dh)

        dhraw = (1 - hs[t] * hs[t]) * dh  # tanh的导数是1-logits^2  dhraw是state 层的导数
        # print("@@@@@@@@ dhraw @@@@@@")
        # print(dhraw)

        dbs += dhraw  # bs-nabla
        dU += np.dot(dhraw, np.transpose(xs[t]))  # U-nabla
        print("@@@@@@@@ dU-%d @@@@@@" % t)
        print(dU)
        if t > 0:
            dW += np.dot(dhraw, np.transpose(hs[t-1]))  # W-nabla
            print("@@@@@@@@ dW-%d @@@@@@" % t)
            print(dW)
        dhnext = dhraw

    return loss, dU, dW, dV, dbs, dbo, hs[seq_length - 1]


# 执行前向+反向传播5次（每次计算的time step为3）

for n in range(5):
  print("######################################################")
  print("#######################第 %d 次 循环###################" % n)
  print("######################################################")
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(x) or n == 0:
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 2 # go from start of data
  inputs =  x[p-2:p+seq_length]
  targets = ground_truth[p:p+seq_length]
  loss, dU, dW, dV, dbs, dbo, hprev = forward_and_backprop(inputs, targets, hprev)
  # perform parameter update with Adagrad
  for param, dparam in zip([U, W, V, bs, bo],
                                [dU, dW, dV, dbs, dbo]):
    param += -learning_rate * dparam #sgd

  p += seq_length # move data pointer

print('U:')
print(U)
print('W:')
print(W)
print('V:')
print(V)


# U:
# [[-0.24492589 -0.23727184 -0.2296178 ]
#  [ 0.39838373  0.49675484  0.59512595]]
# W:
# [[ 0.0992239   0.19968422]
#  [ 0.3000113   0.40001275]]
# V:
# [[ 0.37622149  0.920997  ]
#  [ 0.39517001  0.81996845]
#  [ 0.1286085  -0.54096546]]