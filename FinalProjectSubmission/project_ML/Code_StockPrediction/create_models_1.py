import numpy as np
import csv
from numpy import genfromtxt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Perceptron
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import random
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet

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

data = genfromtxt("C:\CMPSCI_589\Project\Data\\ICICI_train.csv", delimiter = ',')


data_X = data[1:data.shape[0], 1:-1]
data_Y = data[1:data.shape[0], -1]
#train_x, train_y = shuffle_in_unison(data_X, data_Y)
data_x, data_y = shuffle_in_unison(data_X, data_Y)
#print np.random.get_state()
half = data_x.shape[0]/3

# train = data[1:half+1,1:]
# test = data[half+1:data.shape[0]+1,1:]
# #test = data[half+1:half+half,1:]
# print "train data shape:",train.shape
# print "test data shape:",test.shape
# print train[0]
# print test[0]

train_X = data_x[0:half,0:-1]
# print train_X

train_Y = data_y[0:half]
#
#
test_X = data_x[half:,0:-1]
test_Y = data_y[half:]

random.seed(10)
# #================================KNeighborsRegression=================================
# clf = KNeighborsRegressor(n_neighbors = 30)
# clf.fit(train_X,train_Y)
# predictions = clf.predict(test_X)
# MSE = mean_squared_error(test_Y,predictions)
# RMSE = math.sqrt(MSE)
# print RMSE
#
# #===============================Decision Tree Regression==============================
#
# clf = DecisionTreeRegressor(criterion = 'mse')
# clf.fit(train_X,train_Y)
# predictions = clf.predict(test_X)
# MSE = mean_squared_error(test_Y,predictions)
# RMSE = math.sqrt(MSE)
# print RMSE
#
# #===============================Neural Networks==============================
#
clf = MLPRegressor(solver = 'lbfgs', activation = 'tanh', hidden_layer_sizes = (50,), learning_rate='adaptive')
clf.fit(train_X,train_Y)
predictions = clf.predict(test_X)
MSE = mean_squared_error(test_Y,predictions)
RMSE = math.sqrt(MSE)
print RMSE
#
# #===============================Support Vector Regression==============================
# #with linear kernel running indefinitely, with poly giving error "Process finished with exit code ...."
# clf = SVR(kernel = 'rbf')
# clf.fit(train_X,train_Y)
# predictions = clf.predict(test_X)
# MSE = mean_squared_error(test_Y,predictions)
# RMSE = math.sqrt(MSE)
# print RMSE
#
# #==============================Linear Regression=================================
#
# clf = LinearRegression(fit_intercept=True, normalize = True)
# clf.fit(train_X,train_Y)
# predictions = clf.predict(test_X)
# MSE = mean_squared_error(test_Y,predictions)
# RMSE = math.sqrt(MSE)
# print RMSE
#
# #==============================Ridge Regression=================================
#
# clf = Ridge(alpha = 100)
# clf.fit(train_X,train_Y)
# predictions = clf.predict(test_X)
# MSE = mean_squared_error(test_Y,predictions)
# RMSE = math.sqrt(MSE)
# print RMSE
#
# #==============================Lasso Regression=================================
#
# clf = Lasso(alpha = 100, max_iter=100, selection = 'cyclic')
# clf.fit(train_X,train_Y)
# predictions = clf.predict(test_X)
# MSE = mean_squared_error(test_Y,predictions)
# RMSE = math.sqrt(MSE)
# print RMSE

#================================Random Forest Regressor==============================

# clf = RandomForestRegressor(n_estimators = 30, random_state = 10, max_depth = 30, )
# clf.fit(train_X,train_Y)
# predictions = clf.predict(test_X)
# MSE = mean_squared_error(test_Y,predictions)
# RMSE = math.sqrt(MSE)
# print RMSE

#================================Elastic Net==============================

clf = ElasticNet()
clf.fit(train_X,train_Y)
predictions = clf.predict(test_X)
MSE = mean_squared_error(test_Y,predictions)
RMSE = math.sqrt(MSE)
print RMSE