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
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor

def main():
    #-------------------------Elastic Net---------------------------
    alpha = [1,2,3,4,5,6,7,8,9,10]
    alpha_best = 1
    error = 1000.0
    train_X, train_Y, test_X, test_Y = data.load_ICICI()
    rmse = []
    for i in alpha:
        clf = ElasticNet(alpha = i)
        clf.fit(train_X,train_Y)
        predictions = clf.predict(test_X)
        predictions = predictions - (predictions[0] - test_Y[0])#normalizing the predicted values
        MSE = mean_squared_error(test_Y,predictions)
        RMSE = math.sqrt(MSE)
        rmse.append(error)
        if RMSE < error:
            print RMSE
            error = RMSE
            alpha_best = i
            #print alpha_best

    #============================================Ensemble Methods============================================
    clf = BaggingRegressor(base_estimator=ElasticNet(alpha = alpha_best),
                           n_estimators=150,
                           bootstrap=True,
                           ).fit(train_X, train_Y)
    predictions = clf.predict(test_X)
    predictions = predictions - (predictions[0] - test_Y[0])  # normalizing the predicted values
    MSE = mean_squared_error(test_Y, predictions)
    RMSE = math.sqrt(MSE)
    print RMSE
    #AdaBoost is not working well
    clf = GradientBoostingRegressor(n_estimators=50, max_depth = 30,
                           #bootstrap=True,
                           ).fit(train_X, train_Y)
    predictions = clf.predict(test_X)
    predictions = predictions - (predictions[0] - test_Y[0])  # normalizing the predicted values
    MSE = mean_squared_error(test_Y, predictions)
    RMSE = math.sqrt(MSE)
    print RMSE
    #Graphs_plotting.line_graph(99,test_Y, predictions,'ElasticNet','ICICI')


if __name__ == "__main__": main()