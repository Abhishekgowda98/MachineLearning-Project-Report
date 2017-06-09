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


def shuffle_in_unison(a, b, state):
    #randomly shuffles a and b arrays.
    rng_state = state
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a,b

def main():
    #********************************ICICI***********************************************
    train_X, train_Y, test_X, test_Y = data.load_ICICI()
    rmse = []
    #================================KNeighborsRegression=================================
    clf = KNeighborsRegressor(n_neighbors = 30)
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions - (predictions[0] - test_Y[0])#normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "ICICI KNeighborsRegression ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1001,test_Y, predictions,'KNeighborsRegressorBaseline','ICICI')

    # #===============================Decision Tree Regression==============================
    #
    clf = DecisionTreeRegressor(criterion = 'mse')
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions - (predictions[0] - test_Y[0])#normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "ICICI Decision Tree Regression ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1002,test_Y, predictions,'DecisionTreeRegressorBaseline','ICICI')

    #
    # #===============================Neural Networks==============================
    #

    clf = MLPRegressor(solver = 'lbfgs', activation = 'tanh', hidden_layer_sizes = (20,), learning_rate='adaptive')
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions - (predictions[0] - test_Y[0])#normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "ICICI Neural Networks ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1003,test_Y, predictions,'MLPRegressorBaseline','ICICI')

    # #==============================Linear Regression=================================

    clf = LinearRegression(fit_intercept=True, normalize = True)
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions + (predictions[0] - test_Y[0])  # normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "ICICI LinearRegression ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1004,test_Y, predictions,'LinearRegressionBaseline','ICICI')

    # #==============================Ridge Regression=================================
    #
    features = SelectKBest(f_regression)
    clf = Ridge()
    pipeline = Pipeline([('Kbest', features), ('ridge', clf)])
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions + (test_Y[0] + predictions[0])  # normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "ICICI RidgeRegression ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1005,test_Y, predictions,'RidgeBaseline','ICICI')

    clf = RandomForestRegressor(n_estimators = 30, random_state = 10, max_depth = 30)
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions - (predictions[0] - test_Y[0])#normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "ICICI RandomForestRegressor ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1006,test_Y, predictions,'RandomForestRegressorBaseline','ICICI')

    #-------------------------Elastic Net---------------------------
    clf = ElasticNet(alpha = 1)
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions - (predictions[0] - test_Y[0])#normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "ICICI ElasticNet ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1007,test_Y, predictions,'ElasticNetBaseline','ICICI')
    #models_names = ["KNeighborsRegressor", "DecisionTreeRegressor", "LinearRegression", "Ridge", "MLPRegressor", "RandomForestRegressor", "ElasticNet"]
    models_names = ["LR", "DTR", "KNR", "Ridge", "MLPR", "RFR", "EN"]
    Graphs_plotting.bar_chart(10001, rmse, models_names,'ICICIBaseline',7)

    #**********************************************TATA***********************************
    train_X, train_Y, test_X, test_Y = data.load_TATA()
    rmse = []
    #================================KNeighborsRegression=================================
    clf = KNeighborsRegressor(n_neighbors = 30)
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions - (predictions[0] - test_Y[0])#normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "TATA KNeighborsRegression ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1008,test_Y, predictions,'KNeighborsRegressorBaseline','TATA')

    # #===============================Decision Tree Regression==============================
    #
    clf = DecisionTreeRegressor(criterion = 'mse')
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions - (predictions[0] - test_Y[0])#normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "TATA DecisionTreeRegressor ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1009,test_Y, predictions,'DecisionTreeRegressorBaseline','TATA')

    #
    # #===============================Neural Networks==============================
    #

    clf = MLPRegressor(solver = 'lbfgs', activation = 'tanh', hidden_layer_sizes = (20,), learning_rate='adaptive')
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions - (predictions[0] - test_Y[0])#normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "TATA MLPRegressor ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1010,test_Y, predictions,'MLPRegressorBaseline','TATA')
    #

    # #==============================Linear Regression=================================

    clf = LinearRegression(fit_intercept=True, normalize = True)
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions + (predictions[0] - test_Y[0])  # normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "TATA LinearRegression ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1011,test_Y, predictions,'LinearRegressionBaseline','TATA')

    # #==============================Ridge Regression=================================
    #
    features = SelectKBest(f_regression)
    clf = Ridge()
    pipeline = Pipeline([('Kbest', features), ('ridge', clf)])
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions + (test_Y[0] + predictions[0])  # normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "TATA Ridge ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1012,test_Y, predictions,'RidgeBaseline','TATA')


    clf = RandomForestRegressor(n_estimators = 30, random_state = 10, max_depth = 30)
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions - (predictions[0] - test_Y[0])#normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "TATA RandomForestRegressor ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1013,test_Y, predictions,'RandomForestRegressorBaseline','TATA')

    #-------------------------Elastic Net---------------------------
    clf = ElasticNet(alpha = 1)
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions - (predictions[0] - test_Y[0])#normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "TATA ElasticNet ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1014,test_Y, predictions,'ElasticNetBaseline','TATA')
    #models_names = ["KNeighborsRegressor", "DecisionTreeRegressor", "LinearRegression", "Ridge", "MLPRegressor", "RandomForestRegressor", "ElasticNet"]
    models_names = ["LR", "DTR", "KNR", "Ridge", "MLPR", "RFR", "EN"]
    Graphs_plotting.bar_chart(10002, rmse, models_names,'TATABaseline',7)

    #**************************************VEDL*****************************************

    train_X, train_Y, test_X, test_Y = data.load_VEDL()
    rmse = []
    #================================KNeighborsRegression=================================
    clf = KNeighborsRegressor(n_neighbors = 30)
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions - (predictions[0] - test_Y[0])#normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "VEDL KNeighborsRegressor ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1015,test_Y, predictions,'KNeighborsRegressorBaseline','VEDL')

    # #===============================Decision Tree Regression==============================
    #
    clf = DecisionTreeRegressor(criterion = 'mse')
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions - (predictions[0] - test_Y[0])#normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "VEDL DecisionTreeRegressor ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1016,test_Y, predictions,'DecisionTreeRegressorBaseline','VEDL')

    #
    # #===============================Neural Networks==============================
    #

    clf = MLPRegressor(solver = 'lbfgs', activation = 'tanh', hidden_layer_sizes = (20,), learning_rate='adaptive')
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions - (predictions[0] - test_Y[0])#normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "VEDL MLPRegressor ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1017,test_Y, predictions,'MLPRegressorBaseline','VEDL')

    # #==============================Linear Regression=================================

    clf = LinearRegression(fit_intercept=True, normalize = True)
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions + (predictions[0] - test_Y[0])  # normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "VEDL LinearRegression ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1018,test_Y, predictions,'LinearRegressionBaseline','VEDL')

    # #==============================Ridge Regression=================================
    #
    features = SelectKBest(f_regression)
    clf = Ridge()
    pipeline = Pipeline([('Kbest', features), ('ridge', clf)])
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions + (test_Y[0] + predictions[0])  # normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "VEDL Ridge ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1019,test_Y, predictions,'RidgeBaseline','VEDL')


    clf = RandomForestRegressor(n_estimators = 30, random_state = 10, max_depth = 30)
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions - (predictions[0] - test_Y[0])#normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "VEDL RandomForestRegressor ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1020,test_Y, predictions,'RandomForestRegressorBaseline','VEDL')

    #-------------------------Elastic Net---------------------------
    clf = ElasticNet(alpha = 1)
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions - (predictions[0] - test_Y[0])#normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "VEDL ElasticNet ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1021,test_Y, predictions,'ElasticNetBaseline','VEDL')
    #models_names = ["KNeighborsRegressor", "DecisionTreeRegressor", "LinearRegression", "Ridge", "MLPRegressor", "RandomForestRegressor", "ElasticNet"]
    models_names = ["LR", "DTR", "KNR", "Ridge", "MLPR", "RFR", "EN"]
    Graphs_plotting.bar_chart(10003, rmse, models_names,'VEDLBaseline',7)

    #**********************************************REDDY******************************************
    train_X, train_Y, test_X, test_Y = data.load_REDDY()
    rmse = []
    #================================KNeighborsRegression=================================
    clf = KNeighborsRegressor(n_neighbors = 30)
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions - (predictions[0] - test_Y[0])#normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "REDDY KNeighborsRegressor ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1022,test_Y, predictions,'KNeighborsRegressorBaseline','REDDY')

    # #===============================Decision Tree Regression==============================
    #
    clf = DecisionTreeRegressor(criterion = 'mse')
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions - (predictions[0] - test_Y[0])#normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "REDDY DecisionTreeRegressor ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1023,test_Y, predictions,'DecisionTreeRegressorBaseline','REDDY')

    #
    # #===============================Neural Networks==============================
    #

    clf = MLPRegressor(solver = 'lbfgs', activation = 'tanh', hidden_layer_sizes = (20,), learning_rate='adaptive')
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions - (predictions[0] - test_Y[0])#normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "REDDY MLPRegressor ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1024,test_Y, predictions,'MLPRegressorBaseline','REDDY')
    #

    # #==============================Linear Regression=================================

    clf = LinearRegression(fit_intercept=True, normalize = True)
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions + (predictions[0] - test_Y[0])  # normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "REDDY LinearRegression ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1025,test_Y, predictions,'LinearRegressionBaseline','REDDY')

    # #==============================Ridge Regression=================================
    #
    features = SelectKBest(f_regression)
    clf = Ridge()
    pipeline = Pipeline([('Kbest', features), ('ridge', clf)])
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions + (test_Y[0] + predictions[0])  # normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "REDDY Ridge ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1026,test_Y, predictions,'RidgeBaseline','REDDY')

    clf = RandomForestRegressor(n_estimators = 30, random_state = 10, max_depth = 30)
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions - (predictions[0] - test_Y[0])#normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "REDDY RandomForestRegressor ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1027,test_Y, predictions,'RandomForestRegressorBaseline','REDDY')

    #-------------------------Elastic Net---------------------------
    clf = ElasticNet(alpha = 1)
    clf.fit(train_X,train_Y)
    predictions = clf.predict(test_X)
    #predictions = predictions - (predictions[0] - test_Y[0])#normalizing the predicted values
    MSE = mean_squared_error(test_Y,predictions)
    RMSE = math.sqrt(MSE)
    print "REDDY ElasticNet ",RMSE
    rmse.append(RMSE)
    Graphs_plotting.line_graph(1028,test_Y, predictions,'ElasticNetBaseline','REDDY')
    #models_names = ["KNeighborsRegressor", "DecisionTreeRegressor", "LinearRegression", "Ridge", "MLPRegressor", "RandomForestRegressor", "ElasticNet"]
    models_names = ["LR", "DTR", "KNR", "Ridge", "MLPR", "RFR", "EN"]
    Graphs_plotting.bar_chart(10004, rmse, models_names,'REDDYBaseline',7)

if __name__ == "__main__": main()