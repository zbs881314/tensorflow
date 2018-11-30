import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(1)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# parameters
lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28    # MNIST data input
n_steps = 28    # time steps
n_hidden_units = 128   #neurons in hidden layer
n_classes = 10     # MNIST classes

# Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# weights
weights = {
    #(28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    #(128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    #(128,)
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    #(10,)
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

def RNN(X, weights, biases):
    # hidden layer for input to cell
    # transpose the inputs shape from X to (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    
    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in to (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    
    # cell
    
    # basic LSTM Cell
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
        
    # lstm cell is divide into two parts (c_state, h_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    
    # two options for following step
    # 1: tf.nn.rnn(cell, inputs)
    # 2: tf.nn.dynamic_rnn(cell, inputs)
    # if use option 1, you have to modified the shape of X_in, go and check out this:
    # http://github.com/aymericdamien/Tensorflow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
    # in here, we go for option 2
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in
    # make sure the time_major is changed accordingly
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    
    # hidden layer for output as the final results
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']
    
    # unpack to list [(batch, outputs)..] * steps
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))     # state is the last output
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']   # shape=(128, 10)
    
    return results
    
pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs, 
            y: batch_ys, 
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
                x: batch_xs, 
                y: batch_ys, 
                }))
        step += 1
