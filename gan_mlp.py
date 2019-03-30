import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# parameters
batch_size = 256
train_steps = 500
print_step = 10

epsilon = 1e-8
lr_rate = 0.0001
dropout = 0.9
sample_dim = 128
g_hidden = [256, 512, 784]
d_hidden = [784, 256, 1]