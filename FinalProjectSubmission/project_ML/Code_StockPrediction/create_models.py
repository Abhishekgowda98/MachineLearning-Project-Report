import numpy as np
import csv
from numpy import genfromtxt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import math
from sklearn.tree import DecisionTreeRegressor


def shuffle_in_unison(a, b):
    #randomly shuffles a and b arrays.
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a,b
#data = np.load("../DataSets/DRREDDYNS.csv")
#data = open("C:\CMPSCI_589\Project\DataSets\DRREDDYNS.csv",'r')
# data = genfromtxt("C:\CMPSCI_589\Project\DataSets\DRREDDYNS.csv", delimiter = ',')

data = genfromtxt("C:\CMPSCI_589\Project\ProcessedData\\test.csv", delimiter = ',')
print data.shape

data_X = data[1:data.shape[0], 1:-1]
data_Y = data[1:data.shape[0], -1]
train_x, train_y = shuffle_in_unison(data_X, data_Y)

half = data.shape[0]/2
print half
train = data[1:half+1,1:]
test = data[half+1:data.shape[0]+1,1:]
#test = data[half+1:half+half,1:]
print "train data shape:",train.shape
print "test data shape:",test.shape
# print train[0]
# print test[0]

train_X = train[:,0:-1]
train_Y = train[:,-1]
# train_X, train_Y = shuffle_in_unison(train_X, train_Y)
train_x, train_y = shuffle_in_unison(train_X, train_Y)
test_X = test[:,0:-1]
test_Y = test[:,-1]

clf = KNeighborsRegressor(n_neighbors = 5, weights = 'uniform', algorithm = 'auto', verbose = 1)
clf = KNeighborsRegressor(n_neighbors = 100)
clf.fit(train_x,train_y)
predictions = clf.predict(test_X)
print predictions
print test_Y
MSE = mean_squared_error(test_Y,predictions)
RMSE = math.sqrt(MSE)
print RMSE

# clf = DecisionTreeRegressor(criterion = 'mse')
# clf.fit(train_x,train_y)
# predictions = clf.predict(test_X)
# print predictions
# print test_Y
# MSE = mean_squared_error(test_Y,predictions)
# RMSE = math.sqrt(MSE)
# print RMSE
