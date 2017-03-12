#!/usr/bin/python
# 
# CIS 472/572 -- Programming Homework #1
#
# Last modified by: Cathy Webster 3/8/2017
# You are not obligated to use any of this code, but are free to use
# anything you find helpful when completing your assignment.
#
from __future__ import division
import sys
import re
# Node class for the decision tree
#import node
import math
import json
from operator import itemgetter


# SUGGESTED HELPER FUNCTIONS:
# - compute entropy of a 2-valued (Bernoulli) probability distribution 
# - compute information gain for a particular attribute
# - collect counts for each variable value with each class label
# - find the best variable to split on, according to mutual information
# - partition data based on a given variable 


# Load data from a file
def read_data(filename):
  with open(filename) as json_data:
    data = json.load(json_data)
  labels = []
  ingredients = []
  ingredientsDict = {}
  for item in data:
    if item.get('cuisine') not in labels:
      labels.append(item.get('cuisine'))

    for ingredient in item.get('ingredients'):
      if ingredient not in ingredientsDict.keys():
        #ingredients.append(ingredient)
        ingredientsDict[ingredient] = 1
      else:
        ingredientsDict[ingredient] += 1
  lst = sorted(ingredientsDict.iteritems(), key=itemgetter(1), reverse = True)
  i = 0
  for t in lst:
    ingredients.append(t[0])
    i += 1
    if i == 30: break

  examples = []
  for item in data:
    new = [0]*len(ingredients)
    for i in item.get('ingredients'):
      if i in ingredients:
        index = ingredients.index(i)
        new[index] = 1
    new[-1] = labels.index(item.get('cuisine'))
    examples.append(new)

  return (examples), (labels)

def ingredientsCounter(ingredient, ingredientsDict):
  if ingredient not in ingredientsDict.keys():
    ingredientsDict[ingredient] = 1
  else:
    ingredientsDict[ingredient] += 1
  return ingredientsDict

# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
  f = open(modelfile, 'w+')
  root.write(f, 0)

'''
    Author: Cathy Webster
    
    parameters:
        attribute: index of the current attribute, like "plain flour"
        labels: list of possible cuisine names, like "Japanese" "Italian"
    returns:
        count = distribution of [0, 1] for each label
        like, in each example how many occurences of attribute where it's 0
        for each label, how many occurences where it's a 1 for each label
'''
def count(data, attribute, labels):
  count = [[0]*2 for _ in range(len(labels))]
  for d in data:
    if d[attribute] == 0:
      for i in range(len(labels)):
        if d[len(d)-1] == i:
            count[i][0] += 1
    else:
      for i in range(len(labels)):
        if d[len(d)-1] == i:
          count[i][1] += 1
  return count

'''
    Author: Cathy Webster
    
    parameters:
        arr: distribution array, should use result of count
    returns:
        result: entropy
'''
def compEntropy(arr):
  total = 0
  for each in arr:
    total += sum(each)
  result = 0
  for x in arr:
    neg = x[0]
    pos = x[1]
    first = 0
    second = 0
    if neg == 0:
      first = 0
    elif neg != 0:
      first = (-1)*(neg/total)*math.log((neg/total), 2)

    if pos == 0:
      second = 0
    elif pos != 0:
      second = (pos/total)*math.log((pos/total), 2)

    result += first - second

  return result

# Computes information gain for a particular attribute
# Information gain is 
def infoGain(data, attribute, labels):
  # Gets count for the [0,1] distribution of this attribute for each label
  attrCount = count(data, attribute, labels)

  totalZero = 0
  totalOne = 0
  for item in attrCount:
    totalZero += item[0]
    totalOne += item[1]
  total = totalZero + totalOne


  totalZero = attrCount[0] + attrCount[2]
  totalOne = attrCount[1] + attrCount[3]
  firstAttr = attrCount[0] + attrCount[1]
  secAttr = attrCount[2] + attrCount[3]

  dataEntropy = compEntropy(totalZero, totalOne)
  if firstAttr != 0:
    firstAttrEntropy = (firstAttr/total)*compEntropy(attrCount[0], attrCount[1])
  else:
    firstAttrEntropy = 0
  if secAttr != 0:
    secAttrEntropy = (secAttr/total)*compEntropy(attrCount[2], attrCount[3])
  else:
    secAttrEntropy = 0

  result = dataEntropy - firstAttrEntropy - secAttrEntropy
  return result

# Find the best variable to split on, according to mutual information
def bestVar(data, varnames, labels):
  varList = []
  for i,var in enumerate(varnames):
    if i != (len(varnames) - 1):
      if var != "":
        varList.append((infoGain(data, i, labels), i))
  varList.sort(reverse=True)

  return varList[0]

# Partition data based on a given variable 
def partition(data, attribute):
  subsetOne = []
  subsetTwo = []
  for d in data:
    if d[attribute] == 0:
      subsetOne.append(d)
    else:
      subsetTwo.append(d)
  return (subsetOne, subsetTwo)


# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data):
  # >>>> YOUR CODE GOES HERE <<<<
  # Need to get rid of areas where varnames is referenced - don't have set number
  # of attributes/ingredients
  guess = count(data, (len(varnames) - 1))
  if guess[0] > guess[3]:
    g = 0
  else:
    g = 1

  moreFeatures = False 
  for v in varnames:
    if v != "Class" and v != "":
      moreFeatures = True

  if guess[0] == 0 or guess[3] == 0:
    return node.Leaf(varnames, g)
  elif not moreFeatures:
    return node.Leaf(varnames, g)
  else:
    (value, index) = bestVar(data, varnames)
    (no, yes) = partition(data, index)
    var = list(varnames)
    var[index] = ""
    left = build_tree(no, var)
    right = build_tree(yes, var)

    return node.Split(varnames, index, left, right)

# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
  if (len(argv) != 3):
    print 'Usage: id3.py <train> <test> <model>'
    sys.exit(2)
  # "varnames" is a list of names, one for each variable
  # "train" and "test" are lists of examples.  
  # Each example is a list of attribute values, where the last element in
  # the list is the class value.
  (train), (labels) = read_data(argv[0])
  #(test) = read_data(argv[1])
  modelfile = argv[2]

  # build_tree is the main function you'll have to implement, along with
  # any helper functions needed.  It should return the root node of the
  # decision tree.
''' root = build_tree(train)

  print_model(root, modelfile)
  correct = 0
  # The position of the class label is the last element in the list.
  yi = len(test[0]) - 1
  for x in test:
    # Classification is done recursively by the node class.
    # This should work as-is.
    pred = root.classify(x)
    if pred == x[yi]:
      correct += 1
  acc = float(correct)/len(test)
  print "Accuracy: ",acc
'''
if __name__ == "__main__":
  main(sys.argv[1:])
