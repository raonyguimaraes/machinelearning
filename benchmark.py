#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Writing Our First Classifier - Machine Learning Recipes #5
#https://www.youtube.com/watch?v=AoeEHqVSNOw&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal&index=1

from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn import datasets
from sklearn.cross_validation import train_test_split

import numpy as np

def euc(a,b):
	return distance.euclidean(a,b)

class ScrappyKNN():
	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

	def predict(self, X_test):
		predictions = []
		for row in X_test:
			label = self.closest(row)
			predictions.append(label)
		return predictions
	def closest(self, row):
		best_dist = euc(row, self.X_train[0])
		best_index = 0 
		for i in range(1,len(self.X_train)):
			dist = euc(row, self.X_train[i])
			if dist < best_dist:
				best_dist = dist
				best_index = i
		return self.y_train[best_index]


iris = datasets.load_iris()

X = iris.data
y = iris.target


X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = .5)

# from sklearn.neighbors import KNeighborsClassifier
my_classifier = ScrappyKNN()

my_classifier_sklearn = KNeighborsClassifier()


accuracies = []

for i in range (0,1000):
	my_classifier.fit(X_train, y_train)
	predictions = my_classifier.predict(X_test)
	accuracy = accuracy_score(y_test, predictions)
	accuracies.append(accuracy)


print 'ScrappyKNN accuracy mean:', np.mean(accuracies)

accuracies = []

for i in range (0,1000):
	my_classifier_sklearn.fit(X_train, y_train)
	predictions = my_classifier_sklearn.predict(X_test)
	accuracy = accuracy_score(y_test, predictions)
	accuracies.append(accuracy)

print 'sklearn accuracy mean:', np.mean(accuracies)