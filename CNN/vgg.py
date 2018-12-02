import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('tmp/data', one_hot=True)

# parameters
lr_rate = 0.0002
batch_size = 256
train_steps = 1000
print_step = 100
save_step = 500
lr_decay_step = 500

# network parameters
n_input = mnist.train.images.shape[0]
input_dim = mnist.train.images.shape[1]
n_class = 10
dropout = 0.9
n_filters = [32, 64, 128, 256, 256]
n_hidden = [1024, 1024, 10]

def get_placeholder(x, y):
    x = tf.placeholder(tf.float32, shape=[None, x.shape[-1]])
    y = tf.placeholder(tf.float32, shape=[None, y.shape[-1]])
    keep_prob = tf.placeholder(tf.float32)
    lr_rate = tf.placeholder(tf.float32)
    
    return x, y, keep_prob, lr_rate

def conv_one(x, i):
    with tf.variable_scope('conv'+str(i)):
        w = tf.get_variable("w", shape=[3, 3, x.get_shape().as_list()[-1], n_filters[i]],
                                   initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", shape=[n_filters[i]], 
                                    initializer=tf.contrib.layers.xavier_initializer())
        
        conv = tf.nn.conv2d(x, w, strides=[1, 1 , 1, 1], padding='SAME') + b
        output = tf.nn.relu(conv)
    
    
    return output

def conv_two(x, i):
    with tf.variable_scope('conv'+str(i)):
        w1 = tf.get_variable("w1", shape=[3, 3, x.get_shape().as_list()[-1], n_filters[i]])
        b1 = tf.get_variable("b1", shape=[n_filters[i]])
        
        conv1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME') + b1
        
        conv1_output = tf.nn.relu(conv1)
        
        w2 = tf.get_variable("w2", shape=[3, 3, conv1_output.get_shape().as_list()[-1], n_filters[i]])
        b2 = tf.get_variable("b2", shape=[n_filters[i]])
        
        conv2 = tf.nn.conv2d(conv1_output, w2, strides=[1, 1, 1, 1], padding='SAME') + b2
        
        output = tf.nn.relu(conv2)
        
        return output

def max_pool2d(x):
    max_pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    return max_pool