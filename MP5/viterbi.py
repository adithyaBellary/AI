# viterbi.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Renxuan Wang (renxuan2@illinois.edu) on 10/18/2018

import numpy as np

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
	# train_dict = dict([('', dict([('', 0)]))])
	train_dict = dict()

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
	#calculate initial probabilties
	initialProbabilities = dict()
	transitionProbabilities = dict()
	emissionProbabilities = dict()

	POS_Keys = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON',  'DET', 'ADP', 'NUM', 'CONJ','PRT', '.', 'X']
	#initialize dictionaries according to POS_Keys
	for key in POS_Keys:
		initialProbabilities[key] = 0
		transitionProbabilities[key] = {}
		for key1 in POS_Keys:
			transitionProbabilities[key][key1] = 0
		emissionProbabilities[key] = {}
	
	for sentence in train:
		#for each sentence
		
		for word in range(len(sentence)):
			word1 = sentence[word][0]
			tag1 = sentence[word][1]
			if word != (len(sentence)-1):
				tag2 = sentence[word+1][1]

			if word == 0:
				#calculate initial counts
				initialProbabilities[tag1] += 1
			
			if word != (len(sentence)-1):
				#calculate transition counts
				transitionProbabilities[tag1][tag2] += 1 

			#calculate emission counts
			if word1 not in emissionProbabilities[tag1]:
				emissionProbabilities[tag1][word1] = 1
				for key in POS_Keys:
					if key != tag1:
						emissionProbabilities[key][word1] = 0
			else:
				emissionProbabilities[tag1][word1] += 1

	alpha = 1.0
		
	for key in emissionProbabilities:
		s = 0
		for word in emissionProbabilities[key]:
			s += emissionProbabilities[key][word]
		print(key,len(emissionProbabilities[key]), s)

	num_noun_verb_adj = len(emissionProbabilities['NOUN']) + len(emissionProbabilities['VERB']) + len(emissionProbabilities['ADJ'])
	prob_noun = len(emissionProbabilities['NOUN']) / num_noun_verb_adj
	prob_verb = len(emissionProbabilities['VERB']) / num_noun_verb_adj
	prob_adj = len(emissionProbabilities['ADJ']) / num_noun_verb_adj

	POS_choices = ['NOUN', 'VERB', 'ADJ']

	#convert initial counts to log probabilities and smooth
	for key in POS_Keys:
		initialProbabilities[key] = np.log((initialProbabilities[key] + alpha) / (alpha*(len(initialProbabilities) + 1) + len(train)))
		

	#convert transition counts to log probabilities and smooth
	for key in POS_Keys:
		tagCount = 0
		for tag in transitionProbabilities[key]:
			tagCount += transitionProbabilities[key][tag]

		for tag in transitionProbabilities[key]:
			transitionProbabilities[key][tag] = np.log((transitionProbabilities[key][tag] + alpha)  / (tagCount + alpha*(len(transitionProbabilities[key]) + 1)))

	#convert emission counts to log probabilities and smooth
	for key in POS_Keys:
		tagCount = 0
		for tag in emissionProbabilities[key]:
			tagCount += emissionProbabilities[key][tag]

		for tag in emissionProbabilities[key]:
			emissionProbabilities[key][tag] = np.log( (emissionProbabilities[key][tag] + alpha) / (tagCount + alpha * (len(emissionProbabilities[key]) + 1)) )

	# for key in emissionProbabilities:
	# 	s = 0
	# 	for val in emissionProbabilities[key]:
	# 		s += emissionProbabilities[key][val]
	# 	print(s)


	# #sort the dictionaries
	# sortedKeys = sorted(initialProbabilities.keys())
	# temp_init = {}
	# temp_trans = {}
	# temp_em = {}
	# for key in sortedKeys:
	# 	temp_init[key] = initialProbabilities[key]
	# 	temp_trans[key] = transitionProbabilities[key]
	# 	temp_em[key] = emissionProbabilities[key]
	
	# initialProbabilities = temp_init
	# transitionProbabilities = temp_trans
	# emissionProbabilities = temp_em



	numTags = len(POS_Keys)
	bigTrelly = []
	for sentence in test:
		#for each sentence make an empty row for each 
		lilTrelly = []	
		for wordidx in range(len(sentence)):
			tempTrelly = []
			seenFlag = False
			word = sentence[wordidx][0]
			if wordidx == 0:
				#if we are at the first word use the formula
				
				#check to see if we have seent the word before
				for val in emissionProbabilities.values():
					if val.get(sentence[wordidx][0]) != None:
						#we have found the word
						# for tag in initialProbabilities:
						# 	#for each key
						# 	tempTrelly.append(initialProbabilities[tag] + emissionProbabilities[tag][word])
						seenFlag = True
						# lilTrelly.append(tempTrelly)
						break

				if seenFlag:
					#if we have seen the flag yet set the emission probability 
					for tag in POS_Keys:
						# if word in emissionProbabilities[key]:
						# 	emProb = emissionProbabilities[tag][word]
						# else:
						# 	# pred_tag = np.random.choice(POS_choices, 1, p=[prob_noun, prob_verb, prob_adj])
						# 	#is setting it to 0 bad?
						# 	emProb = 0
						# tempTrelly.append(initialProbabilities[tag] + emProb) 	
						tempTrelly.append(initialProbabilities[tag] + emissionProbabilities[word][tag]) 	

					lilTrelly.append(tempTrelly)
				else:
					#if we havent seen the word yet

					####### FIGURE OUT ###################

					for tag in POS_Keys:
						if word in emissionProbabilities[key]:
							emProb = emissionProbabilities[tag][word]
						else:
							# pred_tag = np.random.choice(POS_choices, 1, p=[prob_noun, prob_verb, prob_adj])
							emProb = 0
						tempTrelly.append(initialProbabilities[tag] + emProb) 	
					lilTrelly.append(tempTrelly)

					
			else:
				#for every other word

				#check if we have seen the word or not
				for val in emissionProbabilities.values():
					if val.get(word) != None:
						#we have found the word
						seenFlag = True
						break

				tempTrelly = [0 for i in range(numTags)]
				if wordidx == 1:
					lilTrellyVals = lilTrelly[-1]
				else:
					lilTrellyVals = [i for i, j in lilTrelly[-1]]
				
				for i in range(numTags):
					tempValues = []
					for v in range(numTags):
						if seenFlag:
							x = lilTrellyVals[v] + transitionProbabilities[POS_Keys[v]][POS_Keys[i]] + emissionProbabilities[POS_Keys[i]][word]
						else:
							x = lilTrellyVals[v] + transitionProbabilities[POS_Keys[v]][POS_Keys[i]] ## SMOOTHENED PROB
						tempValues.append(x)
					
					tempTrelly[i] = (max(tempValues), POS_Keys[np.argmax(tempValues)])

				lilTrelly.append(tempTrelly)

		bigTrelly.append(lilTrelly)

	

	# #Trellis creation
	# for sentence in test:
	# 	newTrellis = {}
	# 	oldTrellis = {}
		
	# 	for word in range(len(sentence)):
	# 		seenFlag = False
	# 		if word == 0:
	# 			#if we are on the first word
	# 			for val in emissionProbabilities.values():
	# 				if val.get(sentence[word][0]) != None:
	# 					#we have found the word
	# 					for initialKey in initialProbabilities:
	# 						oldTrellis[initialKey] = initialProbabilities[initialKey] * emissionProbabilities[initialKey][sentence[word][0]]
	# 					seenFlag = True
	# 			if not seenFlag:
	# 				#if we have not seen the word yet
	# 				#optimiza smoothing
	# 				for initialKey in initialProbabilities:
	# 					oldTrellis[initialKey] = initialProbabilities[initialKey] * (alpha / len(initialProbabilities))
				
				
	# 		else:
	# 			#every other word
	# 			for val in emissionProbabilities.values():
	# 				if val.get(sentence[word][0]) != None:
	# 					#we have found the word
						
	# 					seenFlag = True
	# 			if not seenFlag:
	# 				#if we have not seen the word yet
	# 				#optimiza smoothing
	# 				for initialKey in initialProbabilities:
	# 					newTrellis[initialKey] = initialProbabilities[initialKey] * (alpha / len(initialProbabilities))



	return predicts

