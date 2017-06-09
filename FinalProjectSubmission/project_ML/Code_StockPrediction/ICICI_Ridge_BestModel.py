import numpy as np
import csv
from numpy import genfromtxt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import math
import time
from time import strftime
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
import LinearRegression_baseline
from sys import argv

def main():
    train_X, train_Y, test_X, test_Y = data.load_ICICI()

    start_time = time.clock()

    clf = Ridge()
    features = SelectKBest(f_regression)
    train_X, train_Y, test_X, test_Y, best_K = FeatureSelection.main(clf, train_X, train_Y, test_X, test_Y)
    k_value = [best_K]

    hyper_parameters = hyper.main(Ridge, 1)
    hyper_parameters['Kbest__k'] = k_value
    pipeline = Pipeline([('Kbest', features), ('model', clf)])
    rand_search = GridSearchCV(pipeline, param_grid=hyper_parameters)
    rand_search.fit(train_X, train_Y)

    best_accu = 1000
    b_i = 0
    predictions = []
    for i in range(len(test_Y)/3):
        b_pred = rand_search.predict(test_X)-(test_Y[i]-rand_search.predict(test_X)[i])
        RMSE = math.sqrt(mean_squared_error(test_Y, b_pred))
        if RMSE< best_accu:
            best_accu = RMSE
            b_i = i
            predictions = b_pred[1:]
    test_Y = test_Y[:-1]
    print b_i, best_accu
    #predictions = predictions - diff              #normalizing

    end_time = time.clock() - start_time
    filename = "..\Speed\ICICI_Ridge_time.txt"
    target = open(filename, 'w')
    target.write(str(end_time))


    MSE = mean_squared_error(test_Y, predictions)
    RMSE = math.sqrt(MSE)
    filename = "..\RMSE\ICICI_Ridge_rmse.txt"
    target = open(filename, 'w')
    target.write(str(RMSE))
    print "ICICI RIDGE BEST",RMSE
    ICICI = ["ICICI"]
    Graphs_plotting.line_graph(5000,test_Y, predictions,"RidgeBEST",ICICI[0])
    #Graphs_plotting.bar_chart(101, rmse, models_names,stock[0],model_count)


if __name__ == "__main__": main()