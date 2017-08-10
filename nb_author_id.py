#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 
    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

'''
Create and train a Naive Bayes classifier in naive_bayes/nb_author_id.py'

'''

t0 = time()
features_train, features_test, labels_train, labels_test = preprocess()
print "preprocessing time:", round(time()-t0, 3), "s"


t0 = time()

clf = GaussianNB()

clf.fit(features_train, labels_train)

print "The time required to train is :", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "The prediction time is :", round(time()-t0, 3), "s"

accuracy = accuracy_score(labels_test, pred)

print "The accuracy on the test set is: ",accuracy
