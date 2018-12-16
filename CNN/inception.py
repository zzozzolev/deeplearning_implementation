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

def get_placeholder(x, y):
    X = tf.placeholder(tf.float32, [None, x.shape[-1]])
    Y = tf.placeholder(tf.float32, [None, y.shape[-1]])
    keep_prob = tf.placeholder(tf.float32)
    lr_rate_placeholder = tf.placeholder(tf.float32)
    
    return X, Y, keep_prob, lr_rate_placeholder

def conv(x, i):
    with tf.variable_scope("conv"+str(i)):
        w = tf.get_variable("w", [3, 3, x.get_shape().as_list()[-1], n_filters[i]], 
                                                                    initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", [n_filters[i]], initializer=tf.contrib.layers.xavier_initializer())

        c = tf.nn.conv2d(x, w, [1, 1, 1, 1], padding='SAME')

        output = tf.nn.relu(c)
    
    return output

def max_pool(x):
    output = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    print(output)
    
    return output

def inception_naive(input_, i):
    x = tf.identity(input_)
    with tf.variable_scope("inception"+str(i)):
        # 1 x 1
        w1 = tf.get_variable("w1", [1, 1, x.get_shape().as_list()[-1], inception_naive_dim[0]],
                                                                        initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1", [inception_naive_dim[0]], initializer=tf.contrib.layers.xavier_initializer())
        c1 = tf.nn.conv2d(x, w1, [1, 1, 1, 1], padding='SAME') + b1
        o1 = tf.nn.relu(c1)
        print(c1)
        
        # 3 x 3
        w2 = tf.get_variable("w2", [3, 3, x.get_shape().as_list()[-1], inception_naive_dim[1]],
                                                                        initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("b2", [inception_naive_dim[1]], initializer=tf.contrib.layers.xavier_initializer())
        c2 = tf.nn.conv2d(x, w2, [1, 1, 1, 1], padding='SAME') + b2
        o2 = tf.nn.relu(c2)
        print(c2)
        
        # 5 x 5
        w3 = tf.get_variable("w3", [5, 5, x.get_shape().as_list()[-1], inception_naive_dim[2]],
                                                                        initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable("b3", [inception_naive_dim[2]], initializer=tf.contrib.layers.xavier_initializer())
        c3 = tf.nn.conv2d(x, w3, [1, 1, 1, 1], padding='SAME') + b3
        o3 = tf.nn.relu(c3)
        print(c3)
        
        # 3 x 3 max_pooling
        max_pool = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        print(max_pool)
        
        # concat
        concated = tf.concat([o1, o2, o3, max_pool], axis=3)
        print(concated)
        
        return concated

def inception_with_dimension_reduction(input_, i):
    x = tf.identity(input_)
    
    with tf.variable_scope("inception"+str(i)):
        # 1 x 1
        w1 = tf.get_variable("w1", [1, 1, x.get_shape().as_list()[-1], inception_reduction_dim[0]],
                                                                    initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1", [inception_reduction_dim[0]], initializer=tf.contrib.layers.xavier_initializer())
        c1 = tf.nn.conv2d(x, w1, [1, 1, 1, 1], padding="SAME") + b1
        o1 = tf.nn.relu(c1)
        print(o1)
        
        # reduction 3 x 3
        w2_1 = tf.get_variable("w2_1", [1, 1, x.get_shape().as_list()[-1], inception_reduction_dim[1]],
                                                                       initializer=tf.contrib.layers.xavier_initializer())
        b2_1 = tf.get_variable("b2_1", [inception_reduction_dim[1]], initializer=tf.contrib.layers.xavier_initializer())
        c2_1 = tf.nn.conv2d(x, w2_1, [1, 1, 1, 1], padding="SAME") + b2_1
        o2_1 = tf.nn.relu(c2_1)
        
        w2_2 = tf.get_variable("w2_2", [3, 3, o2_1.get_shape().as_list()[-1], inception_reduction_dim[2]],
                                                                       initializer=tf.contrib.layers.xavier_initializer())
        b2_2 = tf.get_variable("b2_2", [inception_reduction_dim[2]], initializer=tf.contrib.layers.xavier_initializer())
        c2_2 = tf.nn.conv2d(o2_1, w2_2, [1, 1, 1, 1], padding="SAME") + b2_2
        o2_2 = tf.nn.relu(c2_2)
        print(o2_2)
        
        # reduction 5 x 5
        w3_1 = tf.get_variable("w3_1", [1, 1, x.get_shape().as_list()[-1], inception_reduction_dim[3]],
                                                                       initializer=tf.contrib.layers.xavier_initializer())
        b3_1 = tf.get_variable("b3_1", [inception_reduction_dim[3]], initializer=tf.contrib.layers.xavier_initializer())
        c3_1 = tf.nn.conv2d(x, w3_1, [1, 1, 1, 1], padding="SAME") + b3_1
        o3_1 = tf.nn.relu(c3_1)
        
        w3_2 = tf.get_variable("w3_2", [5, 5, o3_1.get_shape().as_list()[-1], inception_reduction_dim[4]],
                                                                       initializer=tf.contrib.layers.xavier_initializer())
        b3_2 = tf.get_variable("b3_2", [inception_reduction_dim[4]], initializer=tf.contrib.layers.xavier_initializer())
        c3_2 = tf.nn.conv2d(o3_1, w3_2, [1, 1, 1, 1], padding="SAME") + b3_2
        o3_2 = tf.nn.relu(c3_2)
        print(o3_2)
        
        # reduction maxpool
        pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME")
        w4 = tf.get_variable("w4", [1, 1, pool.get_shape().as_list()[-1], inception_reduction_dim[0]],
                                                                    initializer=tf.contrib.layers.xavier_initializer())
        b4 = tf.get_variable("b4", [inception_reduction_dim[0]], initializer=tf.contrib.layers.xavier_initializer())
        c4 = tf.nn.conv2d(pool, w4, [1, 1, 1, 1], padding="SAME") + b4
        o4 = tf.nn.relu(c4)
        print(o4)
        
        # concat
        concated = tf.concat([o1, o2_2, o3_2, o4], axis=3)
        print(concated)
        
        return concated

def fc(x, i, version='naive'):
    if version == 'naive':
        n_hidden = n_hidden_naive
    else:
        n_hidden = n_hidden_reduction
        
    with tf.variable_scope('fc'+str(i)):
        w = tf.get_variable("w", [x.get_shape().as_list()[-1], n_hidden[i]],
                                                               initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", [n_hidden[i]],
                                   initializer=tf.contrib.layers.xavier_initializer())
        output = tf.matmul(x, w) + b
        print(output)
        
        return output

def network(x, dropout, version='naive'):
    heigth = width = int(np.sqrt(x.shape[1].value))
    x = tf.reshape(x, [-1, width, heigth, 1])
    
    # inception
    for i in range(2):
        x = tf.contrib.layers.layer_norm(x)
        if version == 'naive':
            x = inception_naive(x, i)
        else:
            x = inception_with_dimension_reduction(x, i)
    
    # max_pool
    x = max_pool(x)
    
    # inception
    for i in range(2, 7):
        x = tf.contrib.layers.layer_norm(x)
        if version == 'naive':
            x = inception_naive(x, i)
        else:
            x = inception_with_dimension_reduction(x, i)
    
    # max_pool
    x = max_pool(x)
    
    shape_list = x.get_shape().as_list()
    
    x = tf.reshape(x, [-1, shape_list[1]*shape_list[2]*shape_list[3]])
    
    # fc 
    for i in range(2):
        x = tf.contrib.layers.layer_norm(x)
        x = fc(x, i, 'reduction')
        x = tf.nn.relu(x)
        x = tf.nn.dropout(x, dropout)
    
    # output
    x = fc(x, len(n_hidden)-1)
    
    return x

X, Y, keep_prob, lr_rate_placeholder = get_placeholder(mnist.train.images, mnist.train.labels)

logits = network(X, keep_prob, 'reduction')
preds = tf.nn.softmax(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(lr_rate_placeholder).minimize(loss)

acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1)))) 

# train
init = tf.global_variables_initializer()
lr_rate = 0.0002
with tf.Session() as sess:
    sess.run(init)
    
    for step in range(train_steps):
        x, y = mnist.train.next_batch(batch_size)
        _, c = sess.run([optimizer, loss], feed_dict={X:x, Y:y, keep_prob:dropout, lr_rate_placeholder:lr_rate})
        
        if step % print_step == 0:
            valid_x, valid_y = mnist.test.next_batch(batch_size)
            test_acc = sess.run(acc, feed_dict={X:valid_x, Y:valid_y, keep_prob:1.0, lr_rate_placeholder:lr_rate})
            print("step", step+1)
            print("train_loss:", c)
            print("test_acc:",test_acc)