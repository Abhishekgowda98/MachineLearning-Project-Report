import numpy as np
from subprocess import call
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
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
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold

def main(models, model_count):
    #for i in range (0, model_count):
    if models == LinearRegression:
        hyper_parameters = {}#'model__normalize' : True,

    if models == DecisionTreeRegressor:
        hyper_parameters = {'model__max_features' : ['auto', 'sqrt', 'log2'], 'model__max_depth' : [2, 40]}

    if models == KNeighborsRegressor:
        hyper_parameters = {'model__n_neighbors' : [4,20],}

    if models == Ridge:
        hyper_parameters = {}#'model__alpha' : np.logspace(-10, 10, 5)

    if models == MLPRegressor:
        hyper_parameters = {'model__hidden_layer_sizes' : (50,), 'model__solver' : ['lbfgs'], 'model__activation' : ['identity', 'logistic', 'tanh', 'relu']}#, 'sgd', 'adam'

    if models == RandomForestRegressor:
        hyper_parameters = {'model__max_features': ['auto','log2','sqrt'],'model__n_estimators': [2,50], 'model__max_depth': [2,30],}

    if models == ElasticNet:
        hyper_parameters = {'model__alpha' : [2,3],}#1,2,3,4,5,5,7,8,9,10

    return hyper_parameters

def hyperSelection(models,train_X,train_Y,test_X,test_Y):
    # if models == LinearRegression:
    #
    #
    # if models == DecisionTreeRegressor:
    #
    #
    # if models == KNeighborsRegressor:
    #
    #
    # if models == Ridge:
    #
    #
    # if models == MLPRegressor:
    #
    #
    # if models == RandomForestRegressor:


    if models == ElasticNet:
        #seed = random.seed(20)
        # kf = KFold(n_splits=3)
        # for train_index, test_index in kf.split(train_X):
        #     #print("TRAIN:", train_index, "TEST:", test_index)
        #
        #     X_train, X_test = train_X[train_index], train_X[test_index]
        #     print X_train.shape,X_test.shape
        #     print X_train[0]
        #     y_train, y_test = train_Y[train_index],train_Y[test_index]

        third = len(train_X)/3
        print third
        train_x = train_X[0:2*third,:]
        train_y = train_Y[0:2*third,:]
        test_x = train_X[2*third + 1:,:]
        test_y = train_X[2 * third + 1:,:]

        alpha = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        alpha_best = 1
        error = 1000.0
        #train_X, train_Y, test_X, test_Y = data.load_VEDL()
        rmse = []
        for i in alpha:
            clf = ElasticNet(alpha=i)
            clf.fit(train_X, train_Y)
            predictions = clf.predict(test_X)
            predictions = predictions - (predictions[0] - test_Y[0])  # normalizing the predicted values
            MSE = mean_squared_error(test_Y, predictions)
            RMSE = math.sqrt(MSE)
            rmse.append(error)
            if RMSE < error:
                print RMSE
                error = RMSE
                alpha_best = i
                # print alpha_best

