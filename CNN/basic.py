# import data

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('tmp/data', one_hot=True)

# parameters
lr_rate = 0.0002
batch_size = 256
train_steps = 1000
print_step = 100

# network parameters
n_input = mnist.train.images.shape[0]
input_dim = mnist.train.images.shape[1]
n_class = 10
dropout = 0.9
n_channel = 1
n_filters = [32, 32, 64]
n_hidden = [256, 64, 10]
kernel_size = 3

# graph input
X = tf.placeholder(tf.float32, [None, input_dim])
Y = tf.placeholder(tf.float32, [None, n_class])
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, i):
    with tf.variable_scope('conv'+str(i)):
        w = tf.get_variable("w", shape=[kernel_size, kernel_size, x.get_shape().as_list()[-1], n_filters[i]], 
                                initializer=tf.contr1ib.layers.xavier_initializer())
        b = tf.get_variable("b", shape=[n_filters[i]],
                                initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b
        output = tf.nn.relu(conv)
    
    return output


def maxpool2d(x):
    max_pool = tf.nn.max_pool(x, 
        [1, 2, 2, 1], 
        [1, 2, 2, 1], 
        padding="SAME")
    
    return max_pool


def conv_net(X, dropout):
    height = int(np.sqrt(X.shape[1].value))
    width = height
    X = tf.reshape(X, [-1, height, width, n_channel])


    for i in range(2):
        conv = conv2d(X, i)
        X = maxpool2d(conv)

    shape_list = X.get_shape().as_list()

    reshaped = tf.reshape(X, [-1, shape_list[1]*shape_list[2]*shape_list[3]])

    w = {
        'ffn': tf.Variable(tf.random_normal([shape_list[1]*shape_list[2]*shape_list[3], 256])),
        'out': tf.Variable(tf.random_normal([256, n_class]))
    }
    
    b = {
        'ffn': tf.Variable(tf.random_normal([256])),
        'out': tf.Variable(tf.random_normal([n_class]))
    }
    
    ffn = tf.matmul(reshaped, w['ffn']) + b['ffn']
    out = tf.matmul(ffn, w['out']) + b['out']

    dropout = tf.nn.dropout(out, dropout)

    return dropout

logits = conv_net(X, keep_prob)
pred = tf.nn.softmax(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(lr_rate).minimize(loss)

pre_acc = tf.to_float(tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1)))
acc = tf.reduce_mean(pre_acc)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(train_steps):
        x, y = mnist.train.next_batch(batch_size)
        _, c = sess.run([optimizer, loss], feed_dict={X: x, Y: y, keep_prob:dropout})

        if step % print_step == 0:
            train_acc = sess.run(acc, feed_dict={X: x, Y: y, keep_prob:dropout})
            print("avg loss", c / batch_size)
            print("acc", train_acc)

            valid_x, valid_y = mnist.test.next_batch(batch_size)
            vaild_acc = sess.run(acc, feed_dict={X:valid_x, Y:valid_y, keep_prob:1.0})

            print("valid_acc", valid_acc)