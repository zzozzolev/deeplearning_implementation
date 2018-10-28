# import data

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('tmp/data')

# parameters
lr_rate = 0.001
batch_size = 1024
train_steps = 100
print_step = 10

# network parameters
input_dim = mnist.train.images.shape[1]
n_class = 10
kernel_size = 3
stride = 1

# graph input
X = tf.placeholder(tf.float32, [None, input_dim])
Y = tf.placeholder(tf.float32, [None, n_class])


def conv2d(X, kernel, bias, stride):
    output_size = (X.shape[1] - kernel_size) / stride + 1
    output = tf.Variable(tf.zeros([batch_size, output_size, output_size, 1]))

    for n_batch in range(batch_size):
        for i in range(output_size):
            for j in range(output_size):
                mul = tf.matmul(X[i*stride:kernel_size, j*stride:kernel_size, :], kernel) + b
                acti = tf.nn.relu(mul)
                output = tf.assign(output[n_batch][i][j][0], tf.reduce_sum(acti))
    
    return output


def maxpool2d(X):
    output_size = (X.shape[0] - kernel_size) / stride + 1
    output = tf.Variable(tf.zeros([output_size, output_size, 1]))

    for n_batch in range(batch_size):
        for i in range(output_size):
            for j in range(output_size):
                max_val = tf.argmax(X[i*stride:kernel_size, j*stride:kernel_size, :], axis=0)
                output = tf.assign(output[n_batch][i][j][0], max_val)
    
    return output


def conv_net(X):
    height = np.sqrt(X.shape[1])
    width = height
    channel = 1
    X = tf.reshape(X, [-1, height, width, channel])

    conv_layer1 = maxpool2d(conv2d(X, w['k1'], b['b1'], stride))
    conv_layer2 = maxpool2d(conv2d(conv_layer1, w['k2'], b['b2'], stride))
    
    reshaped = tf.reshape(conv_layer2, [None, 256])

    fnn = tf.matmul(reshaped, w['ffn']) + b['ffn']
    dropout = tf.nn.dropout(fnn)

    return dropout


# weights & biases
w = {
    'k1': tf.Variable(tf.random_normal([None, kernel_size, kernel_size, 1])),
    'k2': tf.Variable(tf.random_normal([None, kernel_size, kernel_size, 1])),
    'ffn': tf.Variable(tf.random_normal([256, n_class]))
}

b = {
    'b1': tf.Variable(tf.random_normal([kernel_size, kernel_size, 1])),
    'b2': tf.Variable(tf.random_normal([kernel_size, kernel_size, 1])),
    'ffn': tf.Variable(tf.random_normal([256]))
}

logits = conv_net(X)
pred = tf.nn.softmax(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, Y))
optimizer = tf.train.AdamOptimizer(lr_rate).minimize(loss)

pre_acc = tf.to_int32(tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1)))
acc = tf.reduce_mean(pre_acc)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in (train_steps):
        x, y = mnist.train.next_batch()
        _, c = sess.run([optimizer, loss], feed_dict={X: x, Y: y})

        if step % print_step == 0:
            acc = sess.run(acc, feed_dict={X: x, Y: y})
            print("avg loss", c / batch_size)
            print("acc", acc)

            valid_x, valid_y = mnist.test.images.next_batch()
            vaild_acc = sess.run(acc, feed_dict={X:valid_x, Y:valid_y})

            print("valid_acc", valid_acc)