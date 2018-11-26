# mp5.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Renxuan Wang (renxuan2@illinois.edu) on 10/18/2018
import sys
import argparse

from reader import load_dataset, strip_tags
from viterbi_EC import viterbi, baseline

"""
This file contains the main application that is run for this MP.
"""

'''
Evaluate output
input:  two lists of sentences with tags on the words
        one is predicted output, one is the correct tags
output: accuracy number (percentage of tags that match)
'''
def compute_accuracies(predicted_sentences, tag_sentences, unseenIdx, seenIdx, seen_acc):
    correct = 0
    incorrect = 0
    count = 0
    # for i in range(len(predicted_sentences)):
    if seen_acc:
        #if we want to check the accuracy of sentences with no unseen words
        idx = seenIdx
    else:
        idx = unseenIdx
    for i in idx:
        for j in range(len(predicted_sentences[i])):
            count += 1
            if predicted_sentences[i][j][1] == tag_sentences[i][j][1]:
                correct += 1
            else:
                incorrect += 1
    return correct/(correct + incorrect)


def main(args):
    train_set = load_dataset(args.training_file, args.case_sensitive)
    test_set = load_dataset(args.test_file, args.case_sensitive)
    if args.baseline:
        print("You are running the baseline algorithm!")
        predTags, unseenIdx, seenIdx = baseline(train_set, strip_tags(test_set))
        check_seen_accuracy = True
        accuracy = compute_accuracies(test_set, predTags, unseenIdx, seenIdx, check_seen_accuracy)
    else: 
        print("You are running the Viterbi algorithm!")
        predTags, unseenIdx, seenIdx = viterbi(train_set, strip_tags(test_set))
        check_seen_accuracy = False
        # check_seen_accuracy = True

        accuracy = compute_accuracies(test_set, predTags, unseenIdx, seenIdx, check_seen_accuracy)
    if check_seen_accuracy:
        print("Accuracy of sentences with no unseen words")
    else:
        print("accuracy of sentences with unseen words")
    print("Accuracy:",accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440 MP5 HMM')
    parser.add_argument('--train', dest='training_file', type=str, 
        help='the file of the training data')
    parser.add_argument('--test', dest='test_file', type=str, 
        help='the file of the testing data')
    parser.add_argument('--case', dest='case_sensitive', default=False, action='store_true', 
        help='Case sensitive (default false)')
    parser.add_argument('--baseline', dest='baseline', default=False, action='store_true', 
        help='Use baseline algorithm')
    parser.add_argument('--viterbi', dest='viterbi', default=False, action='store_true', 
        help='Use Viterbi algorithm')
    args = parser.parse_args()
    if args.training_file == None or args.test_file == None:
        sys.exit('You must specify training file and testing file!')
    if args.baseline ^ args.viterbi == False:
        sys.exit('You must specify using baseline or Viterbi!')

    main(args)
