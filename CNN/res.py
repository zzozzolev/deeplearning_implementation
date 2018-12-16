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

def get_placeholder(x, y):
    X = tf.placeholder(tf.float32, [None, x.shape[-1]])
    Y = tf.placeholder(tf.float32,[None, y.shape[-1]])
    keep_prob = tf.placeholder(tf.float32)
    
    return X, Y, keep_prob

def fc(x, i, dropout):
    with tf.variable_scope('fc'+str(i)):
        x = tf.contrib.layers.layer_norm(x)
        w = tf.get_variable("w", [x.get_shape().as_list()[-1], n_hiddens[i]],
                                   initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", [n_hiddens[i]], initializer=tf.contrib.layers.xavier_initializer())
        x = tf.matmul(x, w) + b
        
        if i < len(n_hiddens)-1:
            x = tf.nn.relu(x)
            print(x)
            x = tf.nn.dropout(x, dropout)
        print(x)
        
        return x

def residual_network(x, i):
    output_dim = str(n_filters[i])
    with tf.variable_scope('{}_{}_{}'.format('conv', output_dim, i)):
        x = tf.identity(x)
        print(x)
        
        c1_stride = (lambda i: 1 if i == 0 else 2)(i)
        w1 = tf.get_variable("w1_"+output_dim, [3, 3, x.get_shape().as_list()[-1], n_filters[i]],
                                        initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1_"+output_dim, [n_filters[i]], initializer=tf.contrib.layers.xavier_initializer())
        c1 = tf.nn.conv2d(x, w1, [1, c1_stride, c1_stride, 1], padding='SAME') + b1
        o1 = tf.nn.relu(c1)
        
        w2 = tf.get_variable("w2_"+output_dim, [3, 3, n_filters[i], n_filters[i]],
                                        initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("b2_"+output_dim, [n_filters[i]], initializer=tf.contrib.layers.xavier_initializer())
        c2 = tf.nn.conv2d(o1, w2, [1, 1, 1, 1], padding='SAME') + b2
        
        # match dimensions
        if i > 0:        
            w3 = tf.get_variable("w_match", [1, 1, x.get_shape().as_list()[-1], n_filters[i]],
                                        initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.get_variable("b_match", [n_filters[i]], initializer=tf.contrib.layers.xavier_initializer())
            c3 = tf.nn.conv2d(x, w3, [1, 2, 2, 1], padding='SAME') + b3
            x = tf.nn.relu(c3)
        
        o2 = c2 + x
        
        x = tf.nn.relu(o2)
        
        print(x)
        
        return x

def residual_bottleneck_network(x, i):
    output_dim = str(n_filters[i])
    with tf.variable_scope('{}_{}_{}'.format('conv', n_filters[i], i)):
        x = tf.identity(x)
        
        c1_stride = (lambda i: 1 if i == 0 else 2)(i)
        w1 = tf.get_variable("w1_"+output_dim, [1, 1, x.get_shape().as_list()[-1], n_filters[i]],
                                        initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1_"+output_dim, [n_filters[i]], initializer=tf.contrib.layers.xavier_initializer())
        c1 = tf.nn.conv2d(x, w1, [1, c1_stride, c1_stride, 1], padding='SAME') + b1
        o1 = tf.nn.relu(c1)
        
        w2 = tf.get_variable("w2_"+output_dim, [3, 3, n_filters[i], n_filters[i]],
                                        initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("b2_"+output_dim, [n_filters[i]], initializer=tf.contrib.layers.xavier_initializer())
        c2 = tf.nn.conv2d(o1, w2, [1, 1, 1, 1], padding='SAME') + b2
        o2 = tf.nn.relu(c2)
        
        w3 = tf.get_variable("w3_"+str(n_filters[i]*4), [1, 1, n_filters[i], n_filters[i]*4],
                                                    initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable("b3_"+str(n_filters[i]*4), [n_filters[i]*4],initializer=tf.contrib.layers.xavier_initializer())
        c3 = tf.nn.conv2d(o2, w3, [1, 1, 1, 1], padding='SAME') + b3
        
        # match dimensions
        if i > 0:        
            w4 = tf.get_variable("w_match", [1, 1, x.get_shape().as_list()[-1], n_filters[i]*4],
                                        initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.get_variable("b_match", [n_filters[i]*4], initializer=tf.contrib.layers.xavier_initializer())
            c4 = tf.nn.conv2d(x, w4, [1, 2, 2, 1], padding='SAME') + b4
            x = tf.nn.relu(c4)
        
        o3 = c3 + x
        
        x = tf.nn.relu(o3)
        print(x)
        
        return x 

def network(x, dropout, version='normal'):
    width = height = int(np.sqrt(x.shape[-1].value))
    
    # reshape
    x = tf.reshape(x, [-1, width, height, 1])
    
    # residual network
    with tf.variable_scope('residual_network'):
        for i in range(len(n_filters)):
            x = tf.contrib.layers.layer_norm(x)
            if version == 'normal':
                x = residual_network(x, i)
            else:
                x = residual_bottleneck_network(x, i)
    
    # avg pool
    pool_size = x.get_shape().as_list()[1]
    x = tf.nn.avg_pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1], padding='SAME')
    print(x)
    
    x_shape = x.get_shape().as_list()
    
    # reshape
    x = tf.reshape(x, [-1, x_shape[1]*x_shape[2]*x_shape[3]])
    
    # fc
    for i in range(len(n_hiddens)):
        x = fc(x, i, dropout)
    
    return x

X, Y, keep_prob = get_placeholder(mnist.train.images, mnist.train.labels)

logits = network(X, keep_prob, 'bottelneck')
preds = tf.nn.softmax(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))
optimizer = tf.train.AdamOptimizer(lr_rate).minimize(loss)

acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(preds, -1), tf.argmax(Y, -1))))