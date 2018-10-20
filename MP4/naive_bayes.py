# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import numpy as np

def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'i like pie' and 'i like cake' in my training set
    Then train_set := [['i','like','pie'], ['i','like','cake']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was spam and second one was ham.
    Then train_labels := [1,0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter you provided with --laplace (1.0 by default)
    """
    # TODO: Write your code here
    # return predicted labels of development set

    #Training
        #Build bag of words model using input emails (train_set)
        #Compute log likelihoods log(P(Word | Type == Spam)
            #Sum likelihoods across all words in email
        #Smooth likelihoods using Laplace smoothing
    
    unigrams_ham, ham_count = bag_of_words(train_set[:int(len(train_set)/2)])
    unigrams_spam, spam_count = bag_of_words(train_set[int(len(train_set)/2):])
 
    # print(unigrams_spam)

    ham_prob = dict([('', 0)])
    spam_prob = dict([('', 0)])

    likelihoods = dict([('',0)])
    
    for word in unigrams_ham:
        ham_prob[word] = np.log((unigrams_ham[word] + smoothing_parameter)/(ham_count + (smoothing_parameter*(len(unigrams_ham)+1))))

    for word in unigrams_spam:
        spam_prob[word] = np.log((unigrams_spam[word] + smoothing_parameter)/(spam_count + (smoothing_parameter*(len(unigrams_spam)+1))))

    #Test
        #MLE classification based on sum of log probabilities

    ham_likelihood = 0
    spam_likelihood = 0
    labels = []

    for email in dev_set:
        ham_likelihood = 0
        spam_likelihood = 0
        
        for word in email:
            if word in ham_prob:
                ham_likelihood += ham_prob[word]
            elif (word not in ham_prob):
                ham_likelihood += np.log(smoothing_parameter/(ham_count + (smoothing_parameter*(len(unigrams_ham)+1))))
            if word in spam_prob:
                spam_likelihood += spam_prob[word]
            elif (word not in spam_prob):
                spam_likelihood += np.log(smoothing_parameter/(spam_count + (smoothing_parameter*(len(unigrams_spam)+1))))
        
        if (ham_likelihood) > (spam_likelihood):
            labels.append(0)
        elif (ham_likelihood) < (spam_likelihood):
            labels.append(1)
    return labels

def bag_of_words(data_set):
    unigrams = dict([('', 0)])
    count = 0
    for email in range(len(data_set)):
        for word in data_set[email]:
            count += 1
            if(word != ('.' or '.\r\n' or ',' or ',\r\n' or '!' or '!\r\n' or '?' or '?\r\n')):
                if(word not in unigrams):
                    unigrams[word] = 1
                else:
                    unigrams[word]+=1

    return unigrams, count