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
k = 10
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

uFactor = tf.reshape(tf.nn.embedding_lookup(U, u), [-1, k])
vFactor = tf.reshape(tf.nn.embedding_lookup(V, v), [-1, k])

merge = tf.concat(1, [uFactor, vFactor, uFactor * vFactor])

# fully connection layer
import math
layer1 = 3 * k
layer2 = 3 * k / 2
scale1 = math.sqrt(6.0 / (layer1 + layer2))
scale2 = math.sqrt(6.0 / (layer2 + 1))

W1 = tf.Variable(tf.random_uniform([layer1, layer2], -scale1, scale1))
b1 = tf.Variable(tf.random_uniform([layer2], -scale1, scale1))
y1 = tf.sigmoid(tf.matmul(merge, W1) + b1)

W2 = tf.Variable(tf.random_uniform([layer2, 1], -scale2, scale2))
b2 = tf.Variable(tf.random_uniform([1], -scale2, scale2))
y  = tf.matmul(y1, W2) + b2

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
        
        trainStep.run(feed_dict={u:batch_u, v:batch_v, r:batch_r})

    # predict
    test_u = testSet[:, 0:1]
    test_v = testSet[:, 1:2]
    test_r = testSet[:, 2:3]

    #predict_r = y.eval(feed_dict={u:test_u, v:test_v, r:test_r})
    #print(test_r[0][0], predict_r[0][0])

    rmse_score = rmse.eval(feed_dict={u:test_u, v:test_v, r:test_r})
    mae_score = mae.eval(feed_dict={u:test_u, v:test_v, r:test_r})
    print("%d/%d\t%.4f\t%.4f"%(epoch+1, epochCount, rmse_score, mae_score))



