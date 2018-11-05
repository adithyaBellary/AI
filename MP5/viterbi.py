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
		

	num_noun = 0
	for word in emissionProbabilities['NOUN']:
		if emissionProbabilities['NOUN'][word]:
			num_noun += emissionProbabilities['NOUN'][word]
	num_verb = 0
	for word in emissionProbabilities['VERB']:
		if emissionProbabilities['VERB'][word]:
			num_verb += emissionProbabilities['VERB'][word]
	num_adj = 0
	for word in emissionProbabilities['ADJ']:
		if emissionProbabilities['ADJ'][word]:
			num_adj += emissionProbabilities['ADJ'][word]
	prob_noun = num_noun / (num_noun + num_adj + num_verb)
	prob_verb = num_verb / (num_noun + num_adj + num_verb)
	prob_adj = num_adj / (num_noun + num_adj + num_verb)

	tagCountDict = {}
	for key in POS_Keys:
		count = 0
		for word in emissionProbabilities[key]:
			if emissionProbabilities[key][word] > 0:
				count += emissionProbabilities[key][word]
		tagCountDict[key] = count

	# print(tagCountDict) 
	# count = 0
	# temp = dict()
	# print(tagCountDict)
	# for sentence in train:
	# 	for word in sentence:
	# 		if(word[1] in temp):
	# 			temp[word[1]] += 1
	# 		else:
	# 			temp[word[1]] = 1
	# print(temp)

	# print(num_noun)
	# print(num_verb)
	# print(num_adj)

	POS_choices = ['NOUN', 'VERB', 'ADJ']

	# print(initialProbabilities)

	#convert initial counts to log probabilities and smooth
	for key in POS_Keys:
		initialProbabilities[key] = np.log(initialProbabilities[key]/len(train))#np.log((initialProbabilities[key] + alpha) / (alpha*(len(POS_Keys) + 1) + len(train)))
	# print(initialProbabilities)	

	#convert transition counts to log probabilities and smooth
	for key in POS_Keys:
		tagCount = 0
		for tag in transitionProbabilities[key]:
			tagCount += transitionProbabilities[key][tag]

		for tag in transitionProbabilities[key]:
			transitionProbabilities[key][tag] = np.log((transitionProbabilities[key][tag] + alpha)  / (tagCount + alpha*(len(POS_Keys) + 1)))

	#convert emission counts to log probabilities and smooth
	for key in POS_Keys:
		# tagCount = 0
		# for tag in emissionProbabilities[key]:
		# 	tagCount += emissionProbabilities[key][tag]

		for tag in emissionProbabilities[key]:
			emissionProbabilities[key][tag] = np.log( (emissionProbabilities[key][tag] + alpha) / (tagCountDict[key] + alpha * (len(emissionProbabilities[key]) + 1)) )

	# s = 0
	# for key in initialProbabilities:
	# 	s += initialProbabilities[key]
	# print("initialProbabilities", s)

	# s = 0
	# for key in transitionProbabilities:
	# 	s = 0
	# 	for key2 in transitionProbabilities[key]:
	# 		s += transitionProbabilities[key][key2]
	# 	print("TransitionProbabilities", s)

	# s = 0
	# for key in emissionProbabilities:
	# 	s = 0
	# 	for key2 in emissionProbabilities[key]:
	# 		s += emissionProbabilities[key][key2]
	# 	print("emissionProbabilities", s)



	numTags = len(POS_Keys)
	bigTrelly = []
	seenCount = 0
	unseenCount = 0
	for sentence in test:
		#for each sentence make an empty row for each 
		lilTrelly = []	
		for wordidx in range(len(sentence)):
			tempTrelly = []
			seenFlag = False
			word = sentence[wordidx]
			# print(word)
			if wordidx == 0:
				#if we are at the first word use the formula
				
				#check to see if we have seent the word before
				for val in emissionProbabilities.values():
					if val.get(sentence[wordidx]) != None:
						#we have found the word

						seenFlag = True
						# lilTrelly.append(tempTrelly)
						break

				if seenFlag:
					#if we have seen the flag yet set the emission probability 
					seenCount += 1

					for tag in POS_Keys:
	
						tempTrelly.append(initialProbabilities[tag] + emissionProbabilities[tag][word]) 	

					lilTrelly.append(tempTrelly)
				else:
					#if we havent seen the word yet

					####### FIGURE OUT ###################
					unseenCount += 1
					for tag in POS_Keys:
						if word in emissionProbabilities[key]:
							emProb = emissionProbabilities[tag][word]
						else:
							# pred_tag = np.random.choice(POS_choices, 1, p=[prob_noun, prob_verb, prob_adj])
							emProb = np.log( (alpha / (tagCountDict[tag] + alpha*(len(POS_Keys) +1) ) )) 
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
							if POS_Keys[i] == 'NOUN':
								x = lilTrellyVals[v] + transitionProbabilities[POS_Keys[v]][POS_Keys[i]] + np.log(prob_noun)
							elif POS_Keys[i] == 'VERB':
								x = lilTrellyVals[v] + transitionProbabilities[POS_Keys[v]][POS_Keys[i]] + np.log(prob_verb)
							elif POS_Keys[i] == 'ADJ':
								x = lilTrellyVals[v] + transitionProbabilities[POS_Keys[v]][POS_Keys[i]] + np.log(prob_adj)
							else:
								x = lilTrellyVals[v] + transitionProbabilities[POS_Keys[v]][POS_Keys[i]] + np.log( (alpha / (tagCountDict[POS_Keys[i]] + alpha*(len(POS_Keys) +1) ) ))

						tempValues.append(x)
					
					tempTrelly[i] = (max(tempValues), POS_Keys[np.argmax(tempValues)])

				lilTrelly.append(tempTrelly)

		bigTrelly.append(lilTrelly)

	# print(len(bigTrelly[0]))
	
	for t in range(len(bigTrelly)):
		trellis = bigTrelly[t]
		#for each sentence
		# print(trellis)
		# print(len(trellis),'\n')
		p = []
		# print(test[t])
		# print(t, len(trellis[-1]),'\n')
		if len(test[t]) > 1:
			most_prob_idx = np.argmax([i for i,j in trellis[-1]])
		if len(test[t]) == 0:
			continue
		if len(test[t]) == 1:
			most_prob_idx = np.argmax(trellis)

		# most_prob_tuple = trellis[-1][most_prob_idx]

		tup = (test[t][-1], POS_Keys[most_prob_idx])
		p.append(tup)

		for i in range((len(trellis)-1), 0, -1 ):
			
			tup = (test[t][i-1], trellis[i][most_prob_idx][1])

			p.append(tup)
			most_prob_idx = POS_Keys.index(trellis[i][most_prob_idx][1])

		p.reverse()
		# print(p,'\n')

		predicts.append(p)
	



	return predicts

