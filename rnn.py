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