import tensorflow as tf
import time
import numpy as np
import copy

# data I/O
# data = open('input.txt', 'r').read().decode('utf-8')
data = u"""观自在菩萨，行深般若波罗蜜多时，照见五蕴皆空，度一切苦厄。
舍利子，色不异空，空不异色，色即是空，空即是色，受想行识，亦复如是。
舍利子，是诸法空相，不生不灭，不垢不净，不增不减。
是故空中无色，无受想行识，无眼耳鼻舌身意，无色声香味触法，无眼界，乃至无意识界。无无明，亦无无明尽，乃至无老死，亦无老死尽。
无苦集灭道，无智亦无得。以无所得故，菩提萨埵，依般若波罗蜜多故，心无罣碍，无罣碍故，无有恐怖，远离颠倒梦想，究竟涅磐。
三世诸佛，依般若波罗蜜多故，得阿耨多罗三藐三菩提。故知般若波罗蜜多，是大神咒，是大明咒，是无上咒，是无等等咒，能除一切苦，真实不虚。
故说般若波罗蜜多咒，即说咒曰：揭谛揭谛波罗揭谛波罗僧揭谛菩提萨婆诃。"""*5

chars = list(set(data))
data_size, num_classes = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, num_classes))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
ix_to_char[num_classes+1] = '<EOF>' #暂时没用上，那应该怎么用？？？
# hyperparameters
state_size = 32 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-3
batch_size = 1
num_classes = num_classes

raw_x = [char_to_ix[ch] for ch in data]
raw_y = [char_to_ix[ch] for ch in data[1:]]
raw_y.append(num_classes-1)

# print(raw_x)
# print(len(raw_x))
# print("@@@@@@@@")
# print(raw_y)
# print(len(raw_y))
# print(data[1:])

def gen_batch():
    # partition raw data into batches and stak them vertically in a data matrix
    batch_partition_length = data_size // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    # do partition
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i: batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i: batch_partition_length * (i + 1)]
    # do epoch
    epoch_size = batch_partition_length // seq_length

    for i in range(epoch_size):
        x = data_x[:, i * seq_length:(i + 1) * seq_length]
        y = data_y[:, i * seq_length:(i + 1) * seq_length]
        yield (x, y)


tf.reset_default_graph()

x = tf.placeholder(tf.int32, [None, None], name='input_placeholder')
y = tf.placeholder(tf.int32, [None, None], name='labels_placeholder')

rnn_inputs = tf.one_hot(x, num_classes)
cell = tf.nn.rnn_cell.BasicLSTMCell(state_size)

# static-->dynamic
# init_state = tf.zeros([batch_size, state_size])
init_state = cell.zero_state(tf.shape(x)[0], dtype=tf.float32)

rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes], initializer=tf.random_normal_initializer(stddev=0.01))
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

logits = tf.reshape(
            tf.matmul(tf.reshape(rnn_outputs, [-1, state_size]), W) + b,
            [batch_size, -1, num_classes])
predictions = tf.nn.softmax(logits)

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

total_loss = tf.reduce_sum(losses)
optimizer = tf.train.AdamOptimizer(learning_rate)
gradients, variables = zip(*optimizer.compute_gradients(total_loss))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)


train_step = optimizer.apply_gradients(zip(gradients, variables))


def train_network(sess, num_steps, state_size=4, verbose=True, training_state=None):
    training_losses = []
    training_loss = 0
    if training_state is None:
        training_state = np.zeros((batch_size, state_size))
    start_time = time.time()
    smooth_loss = -np.log(1.0/num_classes)*seq_length
    for step, (X, Y) in enumerate(gen_batch()):
        if step == 0:
            training_state = sess.run(init_state, feed_dict={x:X})
        tr_losses, training_loss_, training_state, _, predictions_values = \
            sess.run([losses,
                      total_loss,
                      final_state,
                      train_step,
                      predictions],
                     feed_dict={x:X, y:Y, init_state:training_state})
        smooth_loss = smooth_loss*0.999 + training_loss_*0.001
        if step % 100 == 0 and step > 0:
            if verbose:
                print("Average loss at step", step, "for last 100 steps:", smooth_loss, training_loss_)
                end_time = time.time()
                print("time cost:%f" % ((end_time - start_time)*1000))
                start_time = time.time()
                print(u"input:data:%s" % ''.join(ix_to_char[ix] for ix in X[0,:]))
                char_results = np.argmax(predictions_values, axis=-1)
                print(u"predict data:%s" % ''.join(ix_to_char[ix] for ix in char_results.ravel()))
                print(u"ground truth:%s" % ''.join(ix_to_char[ix] for ix in Y[0,:]))

    init_char = ix_to_char[X[0, -1]]
    choice_input = [[char_to_ix[init_char]]]
    result_list = []
    result_list.append(init_char)
    # inference_state_value = training_state.copy()
    inference_state_value = copy.copy(training_state)
    for _ in range(500):
        p, inference_state_value = sess.run([predictions, final_state],
                                            feed_dict={x: choice_input,
                                                       init_state: inference_state_value})
        c = np.argmax(p.ravel())
        choice_input = [[c]]
        result_list.append(ix_to_char[c])
    # print(inference_state_value)
    print("--------predict some text----------\n%s\n++++++++++++" % ''.join(result_list))
    return smooth_loss, training_state

sess = tf.Session()
sess.run(tf.global_variables_initializer())
training_state = np.zeros((batch_size, state_size))

train_epochs = 100
for epoch in range(train_epochs):
    print("epoch : %d" % epoch)
    training_losses, training_state = train_network(sess, seq_length, state_size=state_size, training_state=training_state)


