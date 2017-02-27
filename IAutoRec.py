import tensorflow as tf
import numpy as np
import random
import os

tf.set_random_seed(123456789)
np.random.seed(123456789)
random.seed(123456789)

from collections import defaultdict

import sys
try:
    batchSize = int(sys.argv[1])
    learnRate = float(sys.argv[2])
    gpu = float(sys.argv[3])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
except:
    print("default batchSize learnRate")
    batchSize = 512
    learnRate = 0.05

#batchSize = 32
#learnRate = 0.1

# hyper parameter
k = 250
n_layer = [500, 250, 500]
epochCount = 300
dropout_rate = [0.8, 0.8, 1]
# load data
import data
userCount, itemCount, trainSet, testSet = data.ml_1m(should_shuffle=False)

# train data
trainData = defaultdict(lambda:[0]*itemCount)
trainMask = defaultdict(lambda:[0]*itemCount)
for t in trainSet:
    userId = int(t[0])
    itemId = int(t[1])
    rating = float(t[2])
    trainData[userId][itemId] = rating
    trainMask[userId][itemId] = 1.0

# test data
missCnt = 0
testData = defaultdict(lambda:[0]*itemCount)
testMask = defaultdict(lambda:[0]*itemCount)
for t in testSet:
    userId = int(t[0])
    itemId = int(t[1])
    rating = float(t[2])
    if userId in trainData:
        testData[userId][itemId] = rating
        testMask[userId][itemId] = 1.0
    else:
        missCnt += 1

# evaluate data
allData     = []
allTestData = []
allTestMask = []

for userId in testData:
    allData.append(trainData[userId])
    allTestData.append(testData[userId])
    allTestMask.append(testMask[userId])
allData     = np.array(allData)
allTestData = np.array(allTestData)
allTestMask = np.array(allTestMask)

# auto encoder
data = tf.placeholder(tf.float32, [None, itemCount])
mask = tf.placeholder(tf.float32, [None, itemCount])

import math
scale = math.sqrt(6.0 / (userCount + k))
n_layer = [itemCount] + n_layer + [itemCount]
Ws = []
bs = []
layers = []
y = data
print y
W0 = tf.Variable(tf.random_uniform([itemCount, k], -scale, scale))
b0 = tf.Variable(tf.random_uniform([k], -scale, scale))
W1 = tf.Variable(tf.random_uniform([k, itemCount], -scale, scale))
b1 = tf.Variable(tf.random_uniform([itemCount], -scale, scale))

mid_layer = tf.nn.dropout(tf.nn.softmax(tf.matmul(tf.nn.dropout(data, dropout_rate[0]), W0) + b0), dropout_rate[1])
y = tf.matmul(mid_layer, W1) + b1
y = tf.nn.dropout(y, dropout_rate[2])
"""
for i in xrange(len(n_layer)-1):
    W = tf.Variable(tf.random_uniform([n_layer[i], n_layer[i+1]], -scale, scale))
    b = tf.Variable(tf.random_uniform([n_layer[i+1]], -scale, scale))
    y = tf.nn.dropout(tf.nn.softmax(tf.matmul(y, W) + b), dropout_rate)
    print y
    Ws.append(W)
    bs.append(b)
    layers.append(y)
"""
preData = tf.placeholder(tf.float32, [None, itemCount])
preMask = tf.placeholder(tf.float32, [None, itemCount])
rmse = tf.sqrt(tf.reduce_sum(tf.square((y - preData)*preMask)) / tf.reduce_sum(preMask))

# training
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

loss = tf.reduce_mean(tf.reduce_sum(tf.square((y - data)*mask), 1, keep_dims=True))
trainStep = tf.train.GradientDescentOptimizer(learnRate).minimize(loss)

userIdList = trainData.keys()
for epoch in range(epochCount):
    random.shuffle(userIdList)
    
    # train
    for batchId in range( len(userIdList) / batchSize ):
        start = batchId * batchSize
        end = start + batchSize

        batchData = []
        batchMask = []
        for i in range(start, end):
            userId = userIdList[i]
            batchData.append(trainData[userId])
            batchMask.append(trainMask[userId])

        batchData = np.array(batchData)
        batchMask = np.array(batchMask)
        trainStep.run(feed_dict={data:batchData, mask:batchMask})

    # predict
    eval_rmse = rmse.eval(feed_dict={data:allData, preData:allTestData, preMask:allTestMask})
    loss = 0#loss.eval(feed_dict={data:allData, preData:allTestData, preMask:allTestMask})
    print("epoch %d/%d\tloss: %.4f\trmse: %.4f"%(epoch+1, epochCount, loss, eval_rmse))