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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


def shuffle_in_unison(a, b, state):
    #randomly shuffles a and b arrays.
    rng_state = state
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a,b


#data = np.load("../DataSets/DRREDDYNS.csv")
#data = open("C:\CMPSCI_589\Project\DataSets\DRREDDYNS.csv",'r')
# data = genfromtxt("C:\CMPSCI_589\Project\DataSets\DRREDDYNS.csv", delimiter = ',')

data = genfromtxt("C:\CMPSCI_589\Project\ProcessedData\\test.csv", delimiter = ',')


data_X = data[1:data.shape[0], 1:-1]
data_Y = data[1:data.shape[0], -1]
#train_x, train_y = shuffle_in_unison(data_X, data_Y)
np.random.seed(24)
state = np.random.get_state()
data_x, data_y = shuffle_in_unison(data_X, data_Y, state)
#print np.random.get_state()
half = data_x.shape[0]/3

# train = data[1:half+1,1:]
# test = data[half+1:data.shape[0]+1,1:]
# #test = data[half+1:half+half,1:]
# print "train data shape:",train.shape
# print "test data shape:",test.shape
# print train[0]
# print test[0]

train_X = data_x[0:half,:]
train_Y = data_y[0:half]
test_X = data_x[half:,:]
test_Y = data_y[half:]

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

clf = MLPRegressor(solver = 'lbfgs', activation = 'tanh', hidden_layer_sizes = (10,), learning_rate='adaptive')
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
features = SelectKBest(f_regression)
clf = Ridge()
pipeline = Pipeline([('Kbest', features), ('ridge', clf)])
#grid_search = GridSearchCV(pipeline, {'features': [1,2,3,4,5,6,7,8,9,10], 'ridge_alpha': np.logspace(-10, 10, 5)})
grid_search = GridSearchCV(pipeline, {'Kbest__k': [1,2,3,4,5,6,7,8,9,10], 'ridge__alpha': np.logspace(-10, 10, 5)}, verbose = 1)#, 'ridge_alpha': np.logspace(-10, 10, 5)
grid_search.fit(train_X,train_Y)
predictions = grid_search.predict(test_X)
MSE = mean_squared_error(test_Y,predictions)
RMSE = math.sqrt(MSE)
print RMSE

rand_search = RandomizedSearchCV(pipeline, param_distributions={'Kbest__k': [1,2,3,4,5,6,7,8,9,10], 'ridge__alpha': np.logspace(-10, 10, 5)}, verbose = 1)
rand_search.fit(train_X,train_Y)
predictions = rand_search.predict(test_X)
MSE = mean_squared_error(test_Y,predictions)
RMSE = math.sqrt(MSE)
print RMSE

# #==============================Lasso Regression=================================
#
# clf = Lasso(alpha = 100, max_iter=100, selection = 'cyclic')
# clf.fit(train_X,train_Y)
# predictions = clf.predict(test_X)
# MSE = mean_squared_error(test_Y,predictions)
# RMSE = math.sqrt(MSE)
# print RMSE

#================================Random Forest Regressor==============================

select = SelectKBest(f_regression,k = 10)
clf = RandomForestRegressor(n_estimators = 30, random_state = 10, max_depth = 30 )
clf1 = DecisionTreeRegressor()
pipeline = Pipeline([('kBest',select),('rf',clf)])
rand_search = RandomizedSearchCV(pipeline,
                                 param_distributions = {'kBest__k': [1,2,3,4,5,6,7,8,9,10],
                                                        #'dt__max_features': ['auto','log2','sqrt'],
                                                        'rf__n_estimators': [2,50],
                                                        'rf__max_depth': [2,30],
                                                        })
rand_search.fit(train_X,train_Y)
predictions = rand_search.predict(test_X)
MSE = mean_squared_error(test_Y,predictions)
RMSE = math.sqrt(MSE)
print RMSE