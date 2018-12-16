import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('tmp/data', one_hot=True)

lr_rate = 0.0002
n_class = 10
batch_size = 256
dropout = 0.9

n_filters = [64, 64, 128, 128, 256, 256, 512, 512]
n_hiddens = [1000, n_class]

train_steps = 150
print_step = 10