import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('tmp/data', one_hot=True)

batch_size = 256
dropout = 0.9
n_class = 10

# n_filters
n_filters = [16, 32]
inception_naive_dim = [64, 128, 32]
inception_reduction_dim = [128, 48, 96, 8, 16]

# fc dims
n_hidden_naive = [1024, 1024, n_class]
n_hidden_reduction = [512, 1024, n_class]

lr_rate = 0.0002
train_steps = 150
print_step = 10