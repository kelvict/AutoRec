import tensorflow as tf
import numpy as np

tf.set_random_seed(123456789)
np.random.seed(123456789)

import sys
try:
    batchSize = int(sys.argv[1])
    learnRate = float(sys.argv[2])
except:
    print("batchSize learnRate")
    exit()

print("parameter info:")
print("batch size:\t%d"%batchSize)
print("learn rate:\t%f"%learnRate)
print("="*20)

# hyper parameter
k = 12
epochCount = 100

# load data
import data
userCount, itemCount, trainSet, testSet = data.ml_1m()

print("dataset info:")
print("user count:\t%d"%(userCount))
print("item count:\t%d"%(itemCount))
print("train count:\t%d"%(trainSet.shape[0]))
print("test count:\t%d"%(testSet.shape[0]))
print("="*20)

# embedding layer
u = tf.placeholder(tf.int32,   [None, 1])
v = tf.placeholder(tf.int32,   [None, 1])
r = tf.placeholder(tf.float32, [None, 1])

U = tf.Variable(tf.random_uniform([userCount, k], -0.05, 0.05))
V = tf.Variable(tf.random_uniform([itemCount, k], -0.05, 0.05))

uFactor = tf.nn.embedding_lookup(U, u)
vFactor = tf.nn.embedding_lookup(V, v)

x_image = tf.reshape(tf.batch_matmul(uFactor, vFactor, adj_x=True, adj_y=False), [-1, k, k, 1])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W): 
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# first layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([3 * 3 * 64, 64])
b_fc1 = bias_variable([64])

h_pool2_flat = tf.reshape(h_pool2, [-1, 3 * 3 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([64, 1])
b_fc2 = bias_variable([1])

y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

rmse = tf.sqrt(tf.reduce_mean(tf.square(r - y)))
mae  = tf.reduce_mean(tf.abs(r - y))

# loss function
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

loss = tf.reduce_mean(tf.square(r - y))
trainStep = tf.train.GradientDescentOptimizer(learnRate).minimize(loss)

# iterator
for epoch in range(epochCount):
    np.random.shuffle(trainSet)
 
    # train
    for batchId in range( trainSet.shape[0] / batchSize ):
        start = batchId * batchSize
        end = start + batchSize

        batch_u = trainSet[start:end, 0:1]
        batch_v = trainSet[start:end, 1:2]
        batch_r = trainSet[start:end, 2:3]
        
        trainStep.run(feed_dict={u:batch_u, v:batch_v, r:batch_r, keep_prob:0.5})

    # predict
    test_u = testSet[:, 0:1]
    test_v = testSet[:, 1:2]
    test_r = testSet[:, 2:3]

    #predict_r = y.eval(feed_dict={u:test_u, v:test_v, r:test_r, keep_prob:1.0})
    #print(test_r[0][0], predict_r[0][0])

    rmse_score = rmse.eval(feed_dict={u:test_u, v:test_v, r:test_r})
    mae_score = mae.eval(feed_dict={u:test_u, v:test_v, r:test_r})
    print("%d/%d\t%.4f\t%.4f"%(epoch+1, epochCount, rmse_score, mae_score))



