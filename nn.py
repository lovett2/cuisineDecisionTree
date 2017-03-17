from __future__ import division
from scipy.optimize import minimize
import sys
import re
# Node class for the decision tree
import node
import math
import json
from operator import itemgetter
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix



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
    lst = [(k, ingredientsDict[k]) for k in sorted(ingredientsDict, key=ingredientsDict.get, reverse=True)]
    i = 0
    for t in lst:
        ingredients.append(t[0])
        i += 1
        if i == 1001: break
    examples = []
    output = []
    for item in data:
        #new = [0]*(len(ingredients)+1)
        new = [0]*(len(ingredients))
        for i in item.get('ingredients'):
            if i in ingredients:
                index = ingredients.index(i)
                new[index] = 1

        examples.append(new)
        output.append(labels.index(item.get('cuisine')))
    return (examples), (output)

def main(argv):
    if (len(argv) != 1):
        print ("Usage: id3.py <train>")
        sys.exit(2)
    (examples), (output) = read_data(argv[0])
    print("read data complete.")
    #transform our current dataset to a np array
    #I don't know what np exactly does but sklearn wants us to use it :P
    x = np.array(examples)
    y = np.array(output)

    #split our data into train set and test set, using sklearn method train_test_split.
    #usually, 75% of data goes to train and the rest 25% for test.
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    print("X_train", X_train.shape, "X_test", X_test.shape)

    #there are 4 hidden layers with 10 nodes in each hidden layer
    #this is where we set what we want for our model
    mlp = MLPClassifier(hidden_layer_sizes=(20,10,10))
    print("initialized classifier")

    #create our model using the training data
    mlp.fit(X_train,y_train)
    print("finished fit")

    #make predictions on test set using our model
    predictions = mlp.predict(X_test)
    print("finished predictions")

    #print(confusion_matrix(y_test,predictions))

    #print the accuracy report
    print(classification_report(y_test,predictions))

if __name__ == "__main__":
    main(sys.argv[1:])
