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

def shuffle_in_unison(a, b):
    #randomly shuffles a and b arrays.
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a,b

def main():

    #features = SelectKBest(f_regression)
    k_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    train_X, train_Y, test_X, test_Y = data.load_ICICI()
    clf = ElasticNet(alpha = 3)
    clf.fit(train_X, train_Y)
    predictions = clf.predict(test_X)
    predictions = predictions - (predictions[0] - test_Y[0])
    MSE = mean_squared_error(test_Y, predictions)
    RMSE = math.sqrt(MSE)
    print RMSE

    # # *********************************************TATA data*****************************************
    # train_X, train_Y, test_X, test_Y = data.load_TATA()
    # for i in range(0, model_count):
    #     clf = models[i]()
    #     pipeline = Pipeline([('Kbest', features), ('model', clf)])
    #     rand_search = RandomizedSearchCV(pipeline,
    #                                      param_distributions={'Kbest__k': k_value}, )
    #     # 'ridge__alpha': np.logspace(-10, 10, 5)},
    #     # verbose=1)
    #     rand_search.fit(train_X, train_Y)
    #     predictions = rand_search.predict(test_X)
    #     MSE = mean_squared_error(test_Y, predictions)
    #     RMSE = math.sqrt(MSE)
    #     print RMSE


if __name__ == "__main__": main()