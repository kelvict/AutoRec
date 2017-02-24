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
    batchSize = 32
    learnRate = 0.1

#batchSize = 32
#learnRate = 0.1

# hyper parameter
k = 500
epochCount = 300

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

W1 = tf.Variable(tf.random_uniform([itemCount, k], -scale, scale))
b1 = tf.Variable(tf.random_uniform([k], -scale, scale))
mid = tf.nn.softmax(tf.matmul(data, W1) + b1)

W2 = tf.Variable(tf.random_uniform([k, itemCount], -scale, scale))
b2 = tf.Variable(tf.random_uniform([itemCount], -scale, scale))
y = tf.matmul(mid, W2) + b2

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
    print("epoch %d/%d\trmse: %.4f"%(epoch+1, epochCount, eval_rmse))