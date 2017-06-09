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
import data
import hyper
from subprocess import call
from sklearn.linear_model import ElasticNet
import Graphs_plotting
import FeatureSelection

def main(estimator, train_X,train_Y,test_X,test_Y):
#def main():
    print train_X.shape, train_Y.shape, test_X.shape, test_Y.shape
    best_K = 1
    best_rmse = 1000
    seed = random.seed(20)
    #estimator = ElasticNet()
    for K in range (2,17):
        clf = SelectKBest(f_regression, k = K)
        train_X1 = clf.fit_transform(train_X,train_Y)
        test_X1 = clf.fit_transform(test_X, test_Y)
        estimator.fit(train_X1,train_Y)
        predictions = estimator.predict(test_X1)
        MSE = mean_squared_error(test_Y,predictions)
        RMSE = math.sqrt(MSE)
        if RMSE < best_rmse:
            best_rmse = RMSE
            best_K = K
            #print best_rmse
    clf = SelectKBest(f_regression, k=best_K)
    train_X1 = clf.fit_transform(train_X,train_Y)
    test_X1 = clf.fit_transform(test_X,test_Y)
    print train_X1.shape,train_Y.shape,test_X1.shape,test_Y.shape
    return train_X1,train_Y,test_X1,test_Y,best_K

if __name__ == "__main__": main()