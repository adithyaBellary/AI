# perceptron.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018

"""
This is the main entry point for MP6. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import numpy as np
import time
from mp6 import compute_accuracies
def classify(train_set, train_labels, dev_set, learning_rate,max_iter):
    """
    train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
                This can be thought of as a list of 7500 vectors that are each
                3072 dimensional.  We have 3072 dimensions because there are
                each image is 32x32 and we have 3 color channels.
                So 32*32*3 = 3072
    train_labels - List of labels corresponding with images in train_set
    example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
             and X1 is a picture of a dog and X2 is a picture of an airplane.
             Then train_labels := [1,0] because X1 contains a picture of an animal
             and X2 contains no animals in the picture.

    dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
              It is the same format as train_set
    """
    # TODO: Write your code here
    # return predicted labels of development set
    a = time.time()
    #add the bias term
    ones_ = np.ones(train_set.shape[0]).reshape(-1,1)
    train_set = np.concatenate((train_set,ones_), axis=1)

    ones_ = np.ones(dev_set.shape[0]).reshape(-1,1)
    dev_set = np.concatenate((dev_set,ones_), axis = 1)
    
    #initialize the weights 

    # print('epochs:',max_iter)
    # print('lr:',learning_rate)
    # np.random.seed(100)


    w = np.random.uniform(size= train_set.shape[1])
    for iteration in range(max_iter):
        # np.random.seed(np.random.randint(1000))
        # np.random.seed(10)

        # np.random.shuffle(train_set)
        # np.random.shuffle(train_labels)
        error = 0
        for sample in range( train_set.shape[0]):
            #for each sata sample
            t = np.dot(train_set[sample],w)
            if train_labels[sample]:
                #if there is an animal in the image
                if t < 0:
                    w += learning_rate * train_set[sample]
                    error += 1
            else:
                if t >= 0:
                    w -= learning_rate * train_set[sample]
                    error += 1
        print('error: ',1 - error/train_set.shape[0])
    
    #make predictions on the dev set with our newly trained weights

    # for dev_sample in range(dev_set.shape[0]):
    predictions = [1 if np.dot(i,w) >=0 else 0 for i in dev_set]

    # w_ = np.random.uniform(size= train_set.shape[1])
    # predictions = [1 if np.dot(i,w_) >=0 else 0 for i in dev_set]

    b = time.time()
    print('time:',b-a)
    return predictions

def kNN(trainfeat,trainlabel,testfeat, k):
    #Put your code here
    N = trainfeat.shape[0]
    d = trainfeat.shape[1]
    V = testfeat.shape[0]
    out = np.empty([V])
    distances = np.empty([N])
    for i in range(V):
        for j in range(N):
            # distances[j] = dist.euclidean(testfeat[i],trainfeat[j])
            distances[j] = np.linalg.norm(testfeat[i] - trainfeat[j])
        nnEval = np.argpartition(distances,k)[0:k] #Now have the indices of the training features for the NN classifier
        labelEval = np.empty([k])
        for l in range(k):
            labelEval[l] = trainlabel[nnEval[l]]
        # out[i] = stats.mode(labelEval)[0]
        out[i] = mode(labelEval)[0]

    
    return out

def mode(a, axis=0):
    scores = np.unique(np.ravel(a)) # get ALL unique values
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape)
    oldcounts = np.zeros(testshape)

    for score in scores:
        template = (a == score)
        counts = np.expand_dims(np.sum(template, axis),axis)
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent

    return mostfrequent, oldcounts

def classifyEC(train_set, train_labels, dev_set,learning_rate,max_iter):
    a = time.time()
    out = kNN(train_set, train_labels,dev_set, 10)
    b = time.time()
    print('total time:',b-a)
    return out
