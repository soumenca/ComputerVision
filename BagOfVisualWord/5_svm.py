# Example of kNN implemented from Scratch in Python
from sklearn.metrics import confusion_matrix
import csv
import random
import math
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm

number_claster = 8


def loadDataset(train_filename, test_filename, trainingSet=[], testSet=[]):
    with open(train_filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(number_claster):
                dataset[x][y] = float(dataset[x][y])
            trainingSet.append(dataset[x])

    with open(test_filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(number_claster):
                dataset[x][y] = float(dataset[x][y])
            testSet.append(dataset[x])




def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    temp_list = []
    for x in range(len(testSet)):
        temp_list.append(testSet[x])
        if testSet[x] == predictions[x]:
            correct += 1
    print("The Confusion Matrix is:")
    print(confusion_matrix(temp_list, predictions))
    print(len(testSet))
    return (correct / float(len(testSet))) * 100.0


def main():
    # prepare data
    trainingSet = []
    trainingSet1 = []
    testSet = []
    testSet1 = []
    testLabels = []
    trainLabels = []
    predictions = []
    loadDataset('train_feature_8.csv', 'test_feature_8.csv', trainingSet, testSet)
    print 'Train set: ' + repr(len(trainingSet))
    print 'Test set: ' + repr(len(testSet))

    # generate predictions
    print(len(trainingSet))

    for x in range(len(trainingSet)):
        trainLabels.append(trainingSet[x][-1])

    for x in range(len(trainingSet)):
        trainingSet1.append(trainingSet[x][:-1])

    for x in range(len(testSet)):
        testLabels.append(testSet[x][-1])

    for x in range(len(testSet)):
        testSet1.append(testSet[x][:-1])

    #clf = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='squared_hinge', max_iter=1000, multi_class='ovr', penalty='l2', random_state=None, tol=0.0001, verbose=0)  # make classifier object
    clf=svm.SVC(kernel='linear',gamma=1,C=1.0,probability=True )
    #clf=svm.LinearSVC()
    import time
    t=time.time()
    clf.fit(trainingSet1,trainLabels)
    clf.score(trainingSet1,trainLabels)
    t1=time.time()
    t2=time.time()
    for x in range(len(testSet1)):
        result = clf.predict(testSet1[x])
        result1=clf.decision_function(testSet1[x])
        #result1=clf.predict_proba(testSet1[x])
        print(result1)
        predictions.append(result)
        print('> predicted=' + repr(result[0]) + ', actual=' + repr(testLabels[x]))
    t3 = time.time()
    accuracy = getAccuracy(testLabels, predictions)

    print('Accuracy: ' + repr(accuracy) + '%')
    print("Training time:", (t1 - t))
    print("Testing Time:", (t3 - t2))


main()
