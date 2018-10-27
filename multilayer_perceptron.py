import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# parameters
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
train = mnist.train
valid = mnist.validation
test = mnist.test
lr_rate = 0.0001
n_epochs = 10
batch_size = 1024
print_step = 1

# netword parameters
n_hidden1 = 256
n_hidden2 = 512
n_input = train.images.shape[0]
n_batch = n_input // batch_size
input_size = train.images.shape[1]
n_class = 10

# graph input
X = tf.placeholder(tf.float32, shape=[None, input_size])
Y = tf.placeholder(tf.float32, shape=[None, n_class])

# layers weight & bias
weights = {
    'w1': tf.Variable(tf.random_uniform([input_size, n_hidden1])),
    'w2': tf.Variable(tf.random_uniform([n_hidden1, n_hidden2])),
    'ffn': tf.Variable(tf.random_uniform([n_hidden2, n_class]))
}

biases = {
    'b1': tf.Variable(tf.random_uniform([n_hidden1])),
    'b2': tf.Variable(tf.random_uniform([n_hidden2])),
    'ffn': tf.Variable(tf.random_uniform([n_class]))
}

# create model
def multilayer_perceptron(x):
    layer1 = tf.matmul(x, weights['w1']) + biases['b1']
    layer1 = tf.nn.relu(layer1)
    layer2 = tf.matmul(layer1, weights['w2']) + biases['b2']
    layer2 = tf.nn.relu(layer2)
    output = tf.matmul(layer2, weights['ffn']) + biases['ffn']
    
    return output

# logits
logits = multilayer_perceptron(X)

# define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=lr_rate).minimize(loss)

# initializing the variables
initializer = tf.global_variables_initializer()

# open session
with open(tf.Session()) as sess:
    sess.run(initializer)
    
    # training cycle
    for epoch in range(n_epochs):
        avg_cost = 0
        # loop over all batches
        for i in range(n_batch):
            x, y = train.next_batch(batch_size)
            # run optimization op and cost op
            _, c = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            
            # compute batch avg loss (c is cost of a batch)
            avg_cost += c / n_batch
        if epoch % print_step == 0:
            print("Epoch: {}, cost: {}".format(epoch+1, avg_cost))

    print('training finish')

    # test model
    pred = tf.nn.softmax(logits)
    correct = tf.cast(tf.equal(tf.arg_max(pred, 1), tf.arg_max(Y, 1)), "float")
    accuracy = tf.reduce_mean(correct)
    print("accuray: {}".format(accuracy.eval({X: test.images, Y: test.labels})))