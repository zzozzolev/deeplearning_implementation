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

def plot(samples):
    fig, ax = plt.subplots(1, 10, figsize=(10, 1))
    for i in range(10) :
        ax[i].set_axis_off()
        ax[i].imshow(np.reshape(samples[i], (28, 28)))

    plt.show(fig)
    plt.close(fig)

def get_last_shape(x):
    return x.get_shape().as_list()[-1]

def get_placeholder(x, z_dim):
    x = tf.placeholder(tf.float32, [None, x.shape[1]])
    z = tf.placeholder(tf.float32, [None, z_dim])
    keep_prob = tf.placeholder(tf.float32)
    
    return x, z, keep_prob

def get_random_normal(batch_size, n_dim):
    return np.random.normal(size=[batch_size, n_dim])

def get_random_uniform(batch_size, n_dim):
    return np.random.uniform(size=[batch_size, n_dim])

def mlp(x, keep_prob, n_hidden, model):
    for i in range(len(n_hidden)):
        with tf.variable_scope(model+'/mlp'+str(i)) as scope:
            x = tf.contrib.layers.layer_norm(x, scope=scope)
            w = tf.get_variable("w", [get_last_shape(x), n_hidden[i]], 
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", [n_hidden[i]], 
                                initializer=tf.contrib.layers.xavier_initializer())
            x = tf.matmul(x, w) + b

            if i != len(n_hidden) - 1:
                x = tf.nn.relu(x)
                x = tf.nn.dropout(x, keep_prob)

            print(x)
    
    return x

def generator(z, keep_prob):
    with tf.variable_scope('generator'):
        print(z)
        z = mlp(z, keep_prob, g_hidden, 'generator')
        z = tf.nn.sigmoid(z)        
        return z

def discriminator(x, keep_prob, reuse):
    with tf.variable_scope('discriminator', reuse=reuse) as scope:
        logits = mlp(x, keep_prob, d_hidden, 'discriminator')
        preds = tf.nn.sigmoid(logits)
        return preds

# build graph
tf.reset_default_graph()
X, Z, keep_prob = get_placeholder(mnist.train.images, sample_dim)
fake_X = generator(Z, keep_prob)

d_real = discriminator(X, keep_prob, False)
d_fake = discriminator(fake_X, keep_prob, True)

# log(0)이 되서 무한대로 수렴하는 것을 막기 위해 epsilon을 더해줌
d_loss = -tf.reduce_mean((tf.log(d_real) + tf.log(1 - d_fake)))
g_loss = -tf.reduce_mean(tf.log(d_fake + epsilon))

t_var = tf.trainable_variables()
d_var = [var for var in t_var if 'discriminator' in var.name]
g_var = [var for var in t_var if 'generator' in var.name]

d_optimizer = tf.train.AdamOptimizer(lr_rate).minimize(d_loss, var_list=d_var)
g_optimizer = tf.train.AdamOptimizer(lr_rate).minimize(g_loss, var_list=g_var)

d_real_avg = tf.reduce_mean(d_real)
d_fake_avg = tf.reduce_mean(d_fake)

train_steps = 700000
start_step = 0
print_step = 100
save_step = 1000
lr_rate = 0.0001

# train
save_dir = './checkpoints/gan'
project_name = 'g_hidden{}_d_hidden{}_normal_1d_1g'.format('256_512_784', '784_256_1')
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=2)

with tf.Session() as sess:
    sess.run(init)
    
    for step in range(start_step, train_steps+1):
        if start_step != 0 and step == start_step:
            saver.restore(sess, "{}-{}".format(os.path.join(save_dir, project_name), step))
        
        x, _ = mnist.train.next_batch(batch_size)
        
        z = get_random_normal(batch_size, sample_dim)
        _, d_cost = sess.run([d_optimizer, d_loss], feed_dict={X: x, Z:z, keep_prob:dropout})
        _, g_cost = sess.run([g_optimizer, g_loss], feed_dict={Z:z, keep_prob:dropout})
        
        if step % print_step == 0:
            print("step:", step)
            print("train d_loss: {}, g_loss: {}".format(d_cost, g_cost))
            test_x, _ = mnist.test.next_batch(batch_size)
            
            print("-"*80)
    
            samples = sess.run(fake_X, feed_dict={Z:z, keep_prob:1.0})
            plot(samples)
        
        if step % save_step == 0:
            saver.save(sess, os.path.join(save_dir, project_name), global_step=step)