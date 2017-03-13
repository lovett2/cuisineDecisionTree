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
import node
import math
import json
from operator import itemgetter

MAX_ITER = 10
CUR_ITER = 0

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
        thisIngredient = ingredient.encode('ascii', 'ignore').decode('ascii')
        ingredientsDict[thisIngredient] = 1
      else:
        ingredientsDict[thisIngredient] += 1

  lst = sorted(ingredientsDict.iteritems(), key=itemgetter(1), reverse = True)
  i = 0
  for t in lst:
    ingredients.append(t[0])
    i += 1
    if i == 201: break

  examples = []
  for item in data:
    new = [0]*(len(ingredients)+1)
    for i in item.get('ingredients'):
      if i in ingredients:
        index = ingredients.index(i)
        new[index] = 1
    new[-1] = labels.index(item.get('cuisine'))
    examples.append(new)

  return (examples), (labels), (ingredients)

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
    i = d[len(d)-1]
    # print "i is ", i
    # print d[attribute]
    if d[attribute] == 0:
      count[i][0] = count[i][0] + 1
      # print "count[i][0] is ", count[i][0]
    else:
      count[i][1] = count[i][1] + 1
  return count

'''
    Author: Cathy Webster
    
    parameters:
        arr: distribution array, should use result of count
    returns:
        result: entropy
'''
def compEntropy(distr):
  total = sum(distr)
  result = 0
  #for each value in distr array, compute log
  for x in distr:
    if x == 0:
      result += 0
    else:
      result += (-1)*(x/total)*math.log((x/total),2)

  return result

# Computes information gain for a particular attribute
# Information gain is 
def infoGain(data, attribute, labels):
  # Gets count for the [0,1] distribution of this attribute for each label
  attrCount = count(data, attribute, labels)

  totalZero = []
  totalOne = []
  total = []
  for item in attrCount:
    totalZero.append(item[0])
    totalOne.append(item[1])
    total.append(item[0]+item[1])
  zeroValue = sum(totalZero)
  oneValue = sum(totalOne)
  totalValue = zeroValue + oneValue
  # print zeroValue
  # print oneValue
  if totalValue == 0:
    firstAttrEntropy = 0
    secAttrEntropy = 0
  else:
    firstAttrEntropy = (zeroValue/totalValue)*compEntropy(totalZero)
    secAttrEntropy = (oneValue/totalValue)*compEntropy(totalOne)
  dataEntropy = compEntropy(total)

  return dataEntropy - firstAttrEntropy - secAttrEntropy

# Find the best variable to split on, according to mutual information
def bestVar(data, varnames, labels):
  varList = []
  for i,var in enumerate(varnames):
    # if i != (len(varnames) - 1):
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
def build_tree(data, ingredients, labels):
  # >>>> YOUR CODE GOES HERE <<<<
  # global CUR_ITER
  guess = [0]*(len(labels))
  for item in data:
    index = item[len(item)-1]
    # print index
    # print "Old guess[index] ", guess[index]
    guess[index] = guess[index] + 1
    # print "New guess[index] ", guess[index]
  print guess
  g = 0
  gIndex = 0
  oneLabel = 0

  for item in guess:
    if item > 0:
      oneLabel += 1
    if item > g:
      g = item
      gIndex = guess.index(item)

  moreFeatures = False 
  for i in ingredients:
    if i != "":
      moreFeatures = True
  if oneLabel <= 1:
    print "Only one label", gIndex
    return node.Leaf(ingredients, gIndex)
  elif not moreFeatures:
    return node.Leaf(ingredients, gIndex)
  # elif CUR_ITER == MAX_ITER:
  #   return node.Leaf(ingredients, g)
  else:
    # # CUR_ITER += 1
    # print "Finding best var"
    (value, index) = bestVar(data, ingredients, labels)
    if value <= 0:
      return node.Leaf(ingredients, gIndex)
    (no, yes) = partition(data, index)
    var = list(ingredients)
    var[index] = ""

    left = build_tree(no, var, labels)
    right = build_tree(yes, var, labels)
    return node.Split(ingredients, index, left, right)
    # return node.Leaf(ingredients, g)

# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
  if (len(argv) != 3):
    print 'Usage: id3.py <train> <test> <model>'
    sys.exit(2)
  # "varnames" is a list of names, one for each variable
  # "train" and "test" are lists of examples.  
  # Each example is a list of attribute values, where the last element in
  # the list is the class value.
  (train), (labels), (ingredients) = read_data(argv[0])
  
  fourthTrain = int(len(train)/4)
  threeFourths = len(train) - fourthTrain
  new_train = train[0:threeFourths]
  test = train[threeFourths:len(train)]
  #(test) = read_data(argv[1])
  modelfile = argv[2]

  # build_tree is the main function you'll have to implement, along with
  # any helper functions needed.  It should return the root node of the
  # decision tree.
  root = build_tree(new_train, ingredients, labels)

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

if __name__ == "__main__":
  main(sys.argv[1:])
