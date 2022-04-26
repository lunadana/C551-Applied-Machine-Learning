#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 14:00:26 2022

@author: lunadana
Sources : 
    - https://www.dataquest.io/blog/k-nearest-neighbors-in-python/

1- Setting the value for dependent and independent variables
2 - Splitting the dataset
3 - Fitting the KNN model
4 - Predict the test set

isssues to be resolved : 
    - Both categorical and continuous variables : to solve 
        - Solution 1 (nul) : Treat continuous as categorical, setting threshold
        - Solution 2 : Distance function specific to each variable type
        - Solution 3 (debrouille mais pas optimal) : Assuming we normalize between 0 and 1 
            then we could keep the euclidian for categorical since same order
    - Some continuous variables have different range and order of values, set all of them
    to have the same order by normalizing

"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace         #for debugging 

###########################################################################
#                               HEPATITHIS DATA
###########################################################################

dataset = pd.read_csv("/Users/lunadana/Desktop/COMP551/MiniProject/hepatitis_clean.csv")
dataset = dataset.loc[:, dataset.columns != 'Unnamed: 0']
dataset_predictors = dataset.loc[:, dataset.columns != 'Class']

# data between 1 and 2 for continuous, so that every variable has the same weight
# + we can now use euclidian distance on the whole dataset, treating categorical variables as continuous between 1 and 2:
dataset_predictors_same_range = dataset_predictors.copy()
for i in dataset_predictors_same_range.columns:
    if max(dataset_predictors_same_range[i]) == 2 and min(dataset_predictors_same_range[i]) == 1:
        continue
    dataset_predictors_same_range[i] = 1 + dataset_predictors_same_range[i]/max(dataset_predictors_same_range[i])

# List of continuous variables: "Age", "Bilirubin", "AlkPhosphate", "Sgot", 
# "Albumin", "Protime"
# Threshold: mean(variable) OR median(variable)

dataset_predictors_cat_only = dataset_predictors.copy()
dataset_predictors_cat_only_median = dataset_predictors.copy()

Age_mean = dataset_predictors_cat_only["Age"].mean()
Bili_mean = dataset_predictors_cat_only["Bilirubin"].mean()
Alk_mean = dataset_predictors_cat_only["AlkPhosphate"].mean()
Sgot_mean = dataset_predictors_cat_only["Sgot"].mean()
Albu_mean = dataset_predictors_cat_only["Albumin"].mean()
Prot_mean = dataset_predictors_cat_only["Protime"].mean()

Age_median = dataset_predictors_cat_only["Age"].median()
Bili_median = dataset_predictors_cat_only["Bilirubin"].median()
Alk_median = dataset_predictors_cat_only["AlkPhosphate"].median()
Sgot_median = dataset_predictors_cat_only["Sgot"].median()
Albu_median = dataset_predictors_cat_only["Albumin"].median()
Prot_median = dataset_predictors_cat_only["Protime"].median()

# get dataset with only categorical variables based on the mean
dataset_predictors_cat_only.loc[(dataset_predictors_cat_only.Age < Age_mean), 'Age'] = 1
dataset_predictors_cat_only.loc[(dataset_predictors_cat_only.Age >= Age_mean), 'Age'] = 2

dataset_predictors_cat_only.loc[(dataset_predictors_cat_only.Bilirubin < Bili_mean), 'Bilirubin'] = 1
dataset_predictors_cat_only.loc[(dataset_predictors_cat_only.Bilirubin >= Bili_mean), 'Bilirubin'] = 2

dataset_predictors_cat_only.loc[(dataset_predictors_cat_only.AlkPhosphate < Alk_mean), 'AlkPhosphate'] = 1
dataset_predictors_cat_only.loc[(dataset_predictors_cat_only.AlkPhosphate >= Alk_mean), 'AlkPhosphate'] = 2

dataset_predictors_cat_only.loc[(dataset_predictors_cat_only.Sgot < Sgot_mean), 'Sgot'] = 1
dataset_predictors_cat_only.loc[(dataset_predictors_cat_only.Sgot >= Sgot_mean), 'Sgot'] = 2

dataset_predictors_cat_only.loc[(dataset_predictors_cat_only.Albumin < Albu_mean), 'Albumin'] = 1
dataset_predictors_cat_only.loc[(dataset_predictors_cat_only.Albumin >= Albu_mean), 'Albumin'] = 2

dataset_predictors_cat_only.loc[(dataset_predictors_cat_only.Protime < Prot_mean), 'Protime'] = 1
dataset_predictors_cat_only.loc[(dataset_predictors_cat_only.Protime >= Prot_mean), 'Protime'] = 2

# get dataset with only categorical variables based on the median
dataset_predictors_cat_only_median.loc[(dataset_predictors_cat_only_median.Age < Age_median), 'Age'] = 1
dataset_predictors_cat_only_median.loc[(dataset_predictors_cat_only_median.Age >= Age_median), 'Age'] = 2

dataset_predictors_cat_only_median.loc[(dataset_predictors_cat_only_median.Bilirubin < Bili_median), 'Bilirubin'] = 1
dataset_predictors_cat_only_median.loc[(dataset_predictors_cat_only_median.Bilirubin >= Bili_median), 'Bilirubin'] = 2

dataset_predictors_cat_only_median.loc[(dataset_predictors_cat_only_median.AlkPhosphate < Alk_median), 'AlkPhosphate'] = 1
dataset_predictors_cat_only_median.loc[(dataset_predictors_cat_only_median.AlkPhosphate >= Alk_median), 'AlkPhosphate'] = 2

dataset_predictors_cat_only_median.loc[(dataset_predictors_cat_only_median.Sgot < Sgot_median), 'Sgot'] = 1
dataset_predictors_cat_only_median.loc[(dataset_predictors_cat_only_median.Sgot >= Sgot_median), 'Sgot'] = 2

dataset_predictors_cat_only_median.loc[(dataset_predictors_cat_only_median.Albumin < Albu_median), 'Albumin'] = 1
dataset_predictors_cat_only_median.loc[(dataset_predictors_cat_only_median.Albumin >= Albu_median), 'Albumin'] = 2

dataset_predictors_cat_only_median.loc[(dataset_predictors_cat_only_median.Protime < Prot_median), 'Protime'] = 1
dataset_predictors_cat_only_median.loc[(dataset_predictors_cat_only_median.Protime >= Prot_median), 'Protime'] = 2



###########################################################################
#                               MESIDOR DATA
###########################################################################



dataset2 = pd.read_csv("/Users/lunadana/Desktop/COMP551/MiniProject/messidor_clean.csv")
dataset2 = dataset2.loc[:, dataset2.columns != 'Unnamed: 0']
dataset2['SignofDR'] = dataset2['SignofDR'].astype(int)
dataset2_predictors = dataset2.loc[:, dataset2.columns != 'SignofDR']

# data between 1 and 2 for continuous, so that every variable has the same weight
# + we can now use euclidian distance on the whole dataset, treating categorical variables as continuous between 1 and 2:
dataset2_predictors_same_range = dataset2_predictors.copy()
for i in dataset2_predictors_same_range.columns:
    dataset2_predictors_same_range[i] = 1+dataset2_predictors_same_range[i]/max(dataset2_predictors_same_range[i])

# List of continuous variables: "Age", "Bilirubin", "AlkPhosphate", "Sgot", 
# "Albumin", "Protime"
# Threshold: mean(variable) OR median(variable)

dataset2_predictors_cat_only = dataset2_predictors.copy()
dataset2_predictors_cat_only_median = dataset2_predictors.copy()

MA05_mean = dataset2_predictors_cat_only["MAdetectionCL0.5"].mean()
MA06_mean = dataset2_predictors_cat_only["MAdetectionCL0.6"].mean()
MA07_mean = dataset2_predictors_cat_only["MAdetectionCL0.7"].mean()
MA08_mean = dataset2_predictors_cat_only["MAdetectionCL0.8"].mean()
MA09_mean = dataset2_predictors_cat_only["MAdetectionCL0.9"].mean()
MA1_mean = dataset2_predictors_cat_only["MAdetectionCL1"].mean()
mean_8 = dataset2_predictors_cat_only["8"].mean()
mean_9 = dataset2_predictors_cat_only["9"].mean()
mean_10 = dataset2_predictors_cat_only["10"].mean()
mean_11 = dataset2_predictors_cat_only["11"].mean()
mean_12 = dataset2_predictors_cat_only["12"].mean()
mean_13 = dataset2_predictors_cat_only["13"].mean()
mean_14 = dataset2_predictors_cat_only["14"].mean()
mean_15 = dataset2_predictors_cat_only["15"].mean()
mean_Euc = dataset2_predictors_cat_only["EuclidianDistance"].mean()
mean_Opt = dataset2_predictors_cat_only["OpticDiscDiameter"].mean()

MA05_median = dataset2_predictors_cat_only["MAdetectionCL0.5"].median()
MA06_median = dataset2_predictors_cat_only["MAdetectionCL0.6"].median()
MA07_median = dataset2_predictors_cat_only["MAdetectionCL0.7"].median()
MA08_median = dataset2_predictors_cat_only["MAdetectionCL0.8"].median()
MA09_median = dataset2_predictors_cat_only["MAdetectionCL0.9"].median()
MA1_median = dataset2_predictors_cat_only["MAdetectionCL1"].median()
median_8 = dataset2_predictors_cat_only["8"].median()
median_9 = dataset2_predictors_cat_only["9"].median()
median_10 = dataset2_predictors_cat_only["10"].median()
median_11 = dataset2_predictors_cat_only["11"].median()
median_12 = dataset2_predictors_cat_only["12"].median()
median_13 = dataset2_predictors_cat_only["13"].median()
median_14 = dataset2_predictors_cat_only["14"].median()
median_15 = dataset2_predictors_cat_only["15"].median()
median_Euc = dataset2_predictors_cat_only["EuclidianDistance"].median()
median_Opt = dataset2_predictors_cat_only["OpticDiscDiameter"].median()

# get dataset with only categorical variables based on the mean
dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['MAdetectionCL0.5'] < MA05_mean), 'MAdetectionCL0.5'] = 0
dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['MAdetectionCL0.5'] >= MA05_mean), 'MAdetectionCL0.5'] = 1

dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['MAdetectionCL0.6'] < MA06_mean), 'MAdetectionCL0.6'] = 0
dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['MAdetectionCL0.6'] >= MA06_mean), 'MAdetectionCL0.6'] = 1

dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['MAdetectionCL0.7'] < MA07_mean), 'MAdetectionCL0.7'] = 0
dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['MAdetectionCL0.7'] >= MA07_mean), 'MAdetectionCL0.7'] = 1

dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['MAdetectionCL0.8'] < MA08_mean), 'MAdetectionCL0.8'] = 0
dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['MAdetectionCL0.8'] >= MA08_mean), 'MAdetectionCL0.8'] = 1

dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['MAdetectionCL0.9'] < MA09_mean), 'MAdetectionCL0.9'] = 0
dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['MAdetectionCL0.9'] >= MA09_mean), 'MAdetectionCL0.9'] = 1

dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['MAdetectionCL1'] < MA1_mean), 'MAdetectionCL1'] = 0
dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['MAdetectionCL1'] >= MA1_mean), 'MAdetectionCL1'] = 1

dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['8'] < mean_8), '8'] = 0
dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['8'] >= mean_8), '8'] = 1

dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['9'] < mean_9), '9'] = 0
dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['9'] >= mean_9), '9'] = 1

dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['10'] < mean_10), '10'] = 0
dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['10'] >= mean_10), '10'] = 1

dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['11'] < mean_11), '11'] = 0
dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['11'] >= mean_11), '11'] = 1

dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['12']< mean_12), '12'] = 0
dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['12'] >= mean_12), '12'] = 1

dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['13'] < mean_13), '13'] = 0
dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['13'] >= mean_13), '13'] = 1

dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['14'] < mean_14), '14'] = 0
dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['14'] >= mean_14), '14'] = 1

dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['15'] < mean_15), '15'] = 0
dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['15'] >= mean_15), '15'] = 1

dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['EuclidianDistance'] < mean_Euc), 'EuclidianDistance'] = 0
dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['EuclidianDistance'] >= mean_Euc), 'EuclidianDistance'] = 1

dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['OpticDiscDiameter'] < mean_Opt), 'OpticDiscDiameter'] = 0
dataset2_predictors_cat_only.loc[(dataset2_predictors_cat_only['OpticDiscDiameter'] >= mean_Opt), 'OpticDiscDiameter'] = 1



# get dataset with only categorical variables based on the median
dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['MAdetectionCL0.5'] < MA05_median), 'MAdetectionCL0.5'] = 0
dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['MAdetectionCL0.5'] >= MA05_median), 'MAdetectionCL0.5'] = 1

dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['MAdetectionCL0.6'] < MA06_median), 'MAdetectionCL0.6'] = 0
dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['MAdetectionCL0.6'] >= MA06_median), 'MAdetectionCL0.6'] = 1

dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['MAdetectionCL0.7'] < MA07_median), 'MAdetectionCL0.7'] = 0
dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['MAdetectionCL0.7'] >= MA07_median), 'MAdetectionCL0.7'] = 1

dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['MAdetectionCL0.8'] < MA08_median), 'MAdetectionCL0.8'] = 0
dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['MAdetectionCL0.8'] >= MA08_median), 'MAdetectionCL0.8'] = 1

dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['MAdetectionCL0.9'] < MA09_median), 'MAdetectionCL0.9'] = 0
dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['MAdetectionCL0.9'] >= MA09_median), 'MAdetectionCL0.9'] = 1

dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['MAdetectionCL1'] < MA1_median), 'MAdetectionCL1'] = 0
dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['MAdetectionCL1'] >= MA1_median), 'MAdetectionCL1'] = 1

dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['8'] < median_8), '8'] = 0
dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['8'] >= median_8), '8'] = 1

dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['9'] < median_9), '9'] = 0
dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['9'] >= median_9), '9'] = 1

dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['10'] < median_10), '10'] = 0
dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['10'] >= median_10), '10'] = 1

dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['11'] < median_11), '11'] = 0
dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['11'] >= median_11), '11'] = 1

dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['12']< median_12), '12'] = 0
dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['12'] >= median_12), '12'] = 1

dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['13'] < median_13), '13'] = 0
dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['13'] >= median_13), '13'] = 1

dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['14'] < median_14), '14'] = 0
dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['14'] >= median_14), '14'] = 1

dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['15'] < median_15), '15'] = 0
dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['15'] >= median_15), '15'] = 1

dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['EuclidianDistance'] < median_Euc), 'EuclidianDistance'] = 0
dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['EuclidianDistance'] >= median_Euc), 'EuclidianDistance'] = 1

dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['OpticDiscDiameter'] < median_Opt), 'OpticDiscDiameter'] = 0
dataset2_predictors_cat_only_median.loc[(dataset2_predictors_cat_only_median['OpticDiscDiameter'] >= median_Opt), 'OpticDiscDiameter'] = 1

### DATA FEATURES WITH HIGHEST CORRELATION TO SignofDR:
    # higher that 0.1:
    # all 6 MA
    # 11, 12, 13, 14, 15
    # higher than 0.2:
    # MA0.5, MA0.6, MA0.7

###########################################################################
#                               KNN IMPLEMENTATION
###########################################################################


# Distance definition
euclidean = lambda x1, x2: np.sqrt(np.sum((x1 - x2)**2, axis=-1))
manhattan = lambda x1, x2: np.sum(np.abs(x1 - x2), axis=-1)
hamilton = lambda x1, x2: np.sum(np.where(x1 - x2 != 0, 1, 0), axis=-1)
dummy = lambda x1, x2: np.sum(x1+x2, axis=-1)

# Data
# np.array([tuple(r) for r in dataset[['Age','Protime']].to_numpy()])
x, y = np.array([tuple(r) for r in dataset_predictors_cat_only[['Ascites', 'Albumin', 'Histology', 'Protime', 'Bilirubin']].to_numpy()]), np.array(dataset['Class'])                                   #slices the first two columns or features from the data

#[['MAdetectionCL0.5', '12', '13', '15', '14']]
#[['MAdetectionCL0.5','MAdetectionCL0.6','MAdetectionCL0.7','MAdetectionCL0.8','MAdetectionCL0.9','MAdetectionCL1', '12', '13', '15', '14']]

# [['Ascites', 'Albumin', 'Histology', 'Protime', 'Bilirubin', 'Varices', 'Spiders', 'Malaise' , 'Age', 'LiverBig']]
# [['Ascites', 'Albumin', 'Histology', 'Protime', 'Bilirubin']]

#print the feature shape and classes of dataset 
(N,D), C = x.shape, np.max(y)+1
print(f'instances (N) \t {N} \n features (D) \t {D} \n classes (C) \t {C}')


#visualization of the data
#plt.scatter(x_train[:,0], x_train[:,1], c=y_train, marker='o', label='train')
#plt.scatter(x_test[:,0], x_test[:,1], c=y_test, marker='s', label='test')
#plt.legend()
#plt.ylabel('sepal length')
#plt.xlabel('sepal width')
#plt.show()


# KNN Class
class KNN:
    def __init__(self, K=1, dist_fn= hamilton):
        self.dist_fn = dist_fn
        self.K = K
        return
    def fit(self, x, y):
        ''' Store the training data using this method as it is a lazy learner'''
        self.x = x
        self.y = y
        self.C = np.max(y) +1
        return self
    def predict(self, x_test):
        ''' Makes a prediction using the stored training data and the test data given as argument'''
        num_test = x_test.shape[0]
        #calculate distance between the training & test samples and returns an array of shape [num_test, num_train]
        distances = self.dist_fn(self.x[None,:,:], x_test[:,None,:])
        #ith-row of knns stores the indices of k closest training samples to the ith-test sample 
        knns = np.zeros((num_test, self.K), dtype=int)
        #ith-row of y_prob has the probability distribution over C classes
        y_prob = np.zeros((num_test, self.C))
        for i in range(num_test):
            knns[i,:] = np.argsort(distances[i])[:self.K]
            y_prob[i,:] = np.bincount(self.y[knns[i,:]], minlength=self.C) #counts the number of instances of each class in the K-closest training samples
        #y_prob /= np.sum(y_prob, axis=-1, keepdims=True)
        #simply divide by K to get a probability distribution
        y_prob /= self.K
        return y_prob, knns




K_values = []
K_range = [1, 3, 5, 10, 30]
for K_iter in K_range:
    print('\n')
    print(f'Model with K = {K_iter}.')
    avg_acc = 0
    for i in range(100):
        inds = np.random.permutation(N)                 
                                 #generates an indices array from 0 to N-1 and permutes it 

        #split the dataset into train and test
        x_train, y_train = x[inds[:50]], y[inds[:50]]
        x_test, y_test = x[inds[50:]], y[inds[50:]]

        model = KNN(K=K_iter)
        y_prob, knns = model.fit(x_train, y_train).predict(x_test)
        #print('knns shape:', knns.shape)
        #print('y_prob shape:', y_prob.shape)
        #print(y_prob)
    
        #To get hard predictions by choosing the class with the maximum probability
        y_pred = np.argmax(y_prob,axis=-1)
        #print(y_pred)
        accuracy = np.sum(y_pred == y_test)/y_test.shape[0]
        avg_acc = avg_acc + accuracy
    avg_acc = avg_acc/100
    K_values.append(avg_acc)
    print(f'average accuracy is {avg_acc*100:.2f}.')
    #K_iter += 10

x_lin = np.linspace(1, 20, 20)
print("\nAverage accuracy over values of K:")
print(K_values)
print("\nProportion of Y=2 in the test data:")
print(np.count_nonzero(y_test == 1)/len(y_test))
print("\nProportion of Y=2 in the train data:")
print(np.count_nonzero(y_train == 1)/len(y_train))

corr = dataset.corr()

''' 
We observe by printing the accuracy over the K and the proportion of
CLASS = 2 in the test set that the values are equal from a certain value
of K. 
This is indeed an expected result, since the dataset contains only very few 
CLASS = to 1. Thus, when we start takiong into account more neighbors, 
we come across a problem: most of the neighbors have CLASS = 2, simply
because the majority of the dataset has CLASS = 2 (13/80 have CLASS=1).
Hence, all the test y's are then set to 2, and the accuracy is just the 
proportion of test y's equal to 2.
'''

#boolean array to later slice the indexes of correct and incorrect predictions
correct = y_test == y_pred
incorrect = np.logical_not(correct)

#visualization of the points
#plt.scatter(x_train[:,0], x_train[:,1], c=y_train, marker='o', alpha=.2, label='train')
#plt.scatter(x_test[correct,0], x_test[correct,1], marker='.', c=y_pred[correct], label='correct')
#plt.scatter(x_test[incorrect,0], x_test[incorrect,1], marker='x', c=y_test[incorrect], label='misclassified')

#connect each node to k-nearest neighbours in the training set
#for i in range(x_test.shape[0]):
#    for k in range(model.K):
#        hor = x_test[i,0], x_train[knns[i,k],0]
#        ver = x_test[i,1], x_train[knns[i,k],1]
#        plt.plot(hor, ver, 'k-', alpha=.1)
    
#plt.xlabel('Age')
#plt.ylabel('AlkPhosphate')
#plt.legend()
#plt.show()

plt.plot(x_lin, K_values)
plt.show()


'''

Commented out because only works when run with two features.

################# DECISION BOUNDARIES #################

#we can make the grid finer by increasing the number of samples from 200 to higher value
x0v = np.linspace(np.min(x[:,0]), np.max(x[:,0]), 200)
x1v = np.linspace(np.min(x[:,1]), np.max(x[:,1]), 200)

#to features values as a mesh  
x0, x1 = np.meshgrid(x0v, x1v)
x_all = np.vstack((x0.ravel(),x1.ravel())).T

k=5
model = KNN(K=k)

y_train_prob = np.zeros((y_train.shape[0], C))
y_train_prob[np.arange(y_train.shape[0]), y_train] = 1


#to get class probability of all the points in the 2D grid
y_prob_all, _ = model.fit(x_train, y_train).predict(x_all)

y_pred_all = np.zeros_like(y_prob_all)
y_pred_all[np.arange(x_all.shape[0]), np.argmax(y_prob_all, axis=-1)] = 1

#print(y_train_prob)

#plt.scatter(x_train[:,0], x_train[:,1], c=y_train_prob, marker='o', alpha=1)
#plt.scatter(x_all[:,0], x_all[:,1], c=y_pred_all, marker='.', alpha=0.01)
#plt.ylabel('Protime')
#plt.xlabel('Albumin')
#plt.show()

'''








