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

# load data TrainSet->[(uid, iid, rate)]
import data
userCount, itemCount, trainSet, testSet = data.ml_1m(should_shuffle=False)

# train data
trainData = defaultdict(lambda:[0]*userCount) #Mat #Item * #User
trainMask = defaultdict(lambda:[0]*userCount) #if Mat in item is exist
for t in trainSet:
    userId = int(t[0])
    itemId = int(t[1])
    rating = float(t[2])
    trainData[itemId][userId] = rating
    trainMask[itemId][userId] = 1.0

# test data
missCnt = 0 #  num of item exist in testSet but not in trainSet
testData = defaultdict(lambda:[0]*userCount) #Mat #Item * #User
testMask = defaultdict(lambda:[0]*userCount) #if Mat in item is exist
for t in testSet:
    userId = int(t[0])
    itemId = int(t[1])
    rating = float(t[2])
    if itemId in trainData:
        testData[itemId][userId] = rating
        testMask[itemId][userId] = 1.0
    else:
        missCnt += 1

# evaluate data
allData     = [] #user_vec s of All Item in train data
allTestData = [] #user_vec s of All Item in test data
allTestMask = [] #user_mask_vec s of All Item in test data
for itemId in testData:
    allData.append(trainData[itemId])
    allTestData.append(testData[itemId])
    allTestMask.append(testMask[itemId])
allData     = np.array(allData)
allTestData = np.array(allTestData)
allTestMask = np.array(allTestMask)

# auto encoder
#input #item * #user U AutoRec
data = tf.placeholder(tf.float32, [None, userCount])
mask = tf.placeholder(tf.float32, [None, userCount])

import math
scale = math.sqrt(6.0 / (userCount + k))

W1 = tf.Variable(tf.random_uniform([userCount, k], -scale, scale))
b1 = tf.Variable(tf.random_uniform([k], -scale, scale))
mid = tf.nn.softmax(tf.matmul(data, W1) + b1)

W2 = tf.Variable(tf.random_uniform([k, userCount], -scale, scale))
b2 = tf.Variable(tf.random_uniform([userCount], -scale, scale))
y = tf.matmul(mid, W2) + b2

preData = tf.placeholder(tf.float32, [None, userCount])
preMask = tf.placeholder(tf.float32, [None, userCount])
rmse = tf.sqrt(tf.reduce_sum(tf.square((y - preData)*preMask)) / tf.reduce_sum(preMask))

# training
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

loss = tf.reduce_mean(tf.reduce_sum(tf.square((y - data)*mask), 1, keep_dims=True))
trainStep = tf.train.GradientDescentOptimizer(learnRate).minimize(loss)

itemIdList = trainData.keys()
for epoch in range(epochCount):
    random.shuffle(itemIdList)
    
    # train
    for batchId in range( len(itemIdList) / batchSize ):
        start = batchId * batchSize
        end = start + batchSize

        batchData = []
        batchMask = []
        for i in range(start, end):
            itemId = itemIdList[i]
            batchData.append(trainData[itemId])
            batchMask.append(trainMask[itemId])

        batchData = np.array(batchData)
        batchMask = np.array(batchMask)
        trainStep.run(feed_dict={data:batchData, mask:batchMask})

    # predict
    result = rmse.eval(feed_dict={data:allData, preData:allTestData, preMask:allTestMask})
    print("epoch %d/%d\trmse: %.4f"%(epoch+1, epochCount, result))

