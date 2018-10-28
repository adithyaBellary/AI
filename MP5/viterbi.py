# viterbi.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Renxuan Wang (renxuan2@illinois.edu) on 10/18/2018

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

'''
TODO: implement the baseline algorithm.
input:  training data (list of sentences, with tags on the words)
        test data (list of sentences, no tags on the words)
output: list of sentences, each sentence is a list of (word,tag) pairs. 
        E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
'''
def baseline(train, test):
    predicts = []
    train_dict = dict([('', dict([('', 0)]))])

    # train_dict['the'] = {}
    # train_dict['the']['NOUN'] = 1
    # print(train_dict)

    # train_dict['the']['NOUN'] += 1
    # print(train_dict)

    # # train_dict['the']['VERB'] = {}
    # train_dict['the']['VERB'] = 4
    # print(train_dict)

    # tag = max(train_dict['the'], key = train_dict['the'].get)
    # print(tag)

    for sentence in train:
    	for tag in sentence:
    		# print(tag)
    		if tag[0] not in train_dict:		#if we haven't seen word yet, create new dictionary for POS and count
    			train_dict[tag[0]] = {}
    			train_dict[tag[0]][tag[1]] = 1
    		else:
    			if(tag[1] not in train_dict[tag[0]]):	#if we haven't seen tag for that word yet, add that POS into dictionary
    				train_dict[tag[0]][tag[1]] = 1
    			else:
    				train_dict[tag[0]][tag[1]] += 1

    for sentence in test:
    	temp_list = []
    	for word in sentence:
    		if word in train_dict:
    			tag = max(train_dict[word], key = train_dict[word].get)		#Find key of most frequently occurring POS
    		else:															#assign tag to unknown word
    			tag = 'NOUN'
    		temp_list.append((word, tag))
    	predicts.append(temp_list)
    
    return predicts

'''
TODO: implement the Viterbi algorithm.
input:  training data (list of sentences, with tags on the words)
        test data (list of sentences, no tags on the words)
output: list of sentences with tags on the words
        E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
'''
def viterbi(train, test):
    predicts = []
    return predicts
