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

def classifyEC(train_set, train_labels, dev_set,learning_rate,max_iter):
    # Write your code here if you would like to attempt the extra credit
    N = train_set.shape[0]
    n_input = train_set.shape[1]
    n_hidden = 500
    #one for each class
    n_output = 2
    parameters = {}
    y = [1 if label else 0 for label in train_labels]
    y_vec = np.zeros((train_set.shape[0],2))
    for i in range(train_set.shape[0]):
        y_vec[i][y[i]] = 1
    W1 = np.random.randn(n_input,n_hidden)
    W2 = np.random.randn(n_hidden,n_output)
    b1 = np.zeros(shape=(1, n_hidden))
    b2 = np.zeros(shape=(1, n_output))

    for itrations in range(max_iter):
        #make foreard pass

        z1 = np.matmul(train_set,W1) + b1
        a1 = np.tanh(z1)
        z2 = np.matmul(a1, W2) + b2
        # print('z2 shape:',z2.shape)
        scores = np.exp(z2)
        # scores = z2
        print(scores)
        probabilities = scores / np.sum(scores, axis=1, keepdims=True)
        print(probabilities)


        #backpropagation
        
        d3 = probabilities - y_vec

        # print(probabilities.shape)
        # print(y_vec.shape)
        # print(d3.shape)

        dL_dW2 = np.matmul(a1.T,d3)
        dL_db2 = np.sum(d3, axis=0, keepdims=True)
        d2 = np.matmul(d3, W2.T) * (1 - z1**2)
        dL_dW1 = np.matmul(train_set.T, d2)
        dL_db1 = np.sum(d2, axis=0)

        reg_lambda = 0.01

        dL_dW2 += reg_lambda * W2
        dL_dW1 += reg_lambda * W1

        #update gradients
        W1 += -learning_rate * dL_dW1
        W2 += -learning_rate * dL_dW2
        b1 += -learning_rate * dL_db1
        b2 += -learning_rate * dL_db2

        paramaters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


        #check accuracy
        preds_ =  [make_prediction(paramaters, x) for x in dev_set]
        # compute_accuracies(preds_, dev_set, dev_labels)

    return [make_prediction(parameters,x) for x in dev_set]

def make_prediction(parameters, x):
    W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2']
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)
