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