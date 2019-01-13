import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('tmp/data', one_hot=True)

batch_size = 512
lr_rate = 0.0002
train_steps = 1000
print_step = 10

n_layer = 4
time_steps = 28
num_units = 256
n_hidden = num_units * 2
n_class = 10
dropout = 0.9

def get_placeholder(x, y):
    X = tf.placeholder(tf.float32, [None, x.shape[-1]])
    Y = tf.placeholder(tf.float32,[None, y.shape[-1]])
    keep_prob = tf.placeholder(tf.float32)
    lr_rate_placeholder = tf.placeholder(tf.float32)
    
    return X, Y, keep_prob, lr_rate_placeholder

def fc(x, n_hidden, keep_prob):
    with tf.variable_scope('fc'+str(n_hidden)):
        x = tf.contrib.layers.layer_norm(x)
        w = tf.get_variable("w", [x.get_shape().as_list()[-1], n_hidden],
                                   initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", [n_hidden], initializer=tf.contrib.layers.xavier_initializer())
            
        output = tf.matmul(x, w) + b
        
        if n_hidden != n_class:
            output = tf.nn.dropout(output, keep_prob)
        
        print(output)
        
        return output

def rnn(x, n_layer, time_steps, num_units, keep_prob):
    with tf.variable_scope('rnn'):
        x = tf.contrib.layers.layer_norm(x)
        
        cells = []
        for _ in range(n_layer):
            cell = tf.contrib.rnn.BasicLSTMCell(num_units)
            cell = tf.contrib.rnn.DropoutWrapper(cell, keep_prob)
            cells.append(cell)
        
        multi_cells = tf.contrib.rnn.MultiRNNCell(cells)
        
        # [batch_size, sequence_length, num_units]
        outputs, _ = tf.nn.dynamic_rnn(
            multi_cells, x, dtype=tf.float32) 
        
        return outputs

def network(x, n_layer, num_steps, num_units, keep_prob):
    
    x = tf.reshape(x, [-1, 28, 28])
    
    rnn_outputs = rnn(x, n_layer, num_steps, num_units, keep_prob)
    print(rnn_outputs)
    
    fc_input = tf.reshape(rnn_outputs, [-1, num_units])
    print(fc_input)
    
    fc_output = fc(fc_input, n_hidden, keep_prob)
    activated = tf.nn.relu(fc_output)
    print(activated)
    
    output = fc(activated, n_class, keep_prob)
    
    reshaped = tf.reshape(output, [-1, num_steps, n_class])
    print(reshaped)
    
    # [num_steps, batch_size, num_units]
    transposed = tf.transpose(reshaped, [1, 0, 2])
    print(transposed)
    
    last_step_outputs = transposed[num_steps-1]
    print(last_step_outputs)
    
    return last_step_outputs

# train and test
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for step in range(train_steps+1):
        x, y = mnist.train.next_batch(batch_size)
        _, c = sess.run([optimizer, loss], feed_dict={X: x, Y:y, keep_prob:dropout, lr_rate_placeholder:lr_rate})
        
        if step % print_step == 0:
            print("steps:", step)
            print("train_loss:", c)
            
            valid_x, valid_y = mnist.test.next_batch(batch_size)
            valid_acc = sess.run(acc, feed_dict={X:valid_x, Y:valid_y, keep_prob:1.0})
            print("test_acc:", valid_acc)