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

def shuffle_in_unison(a, b):
    #randomly shuffles a and b arrays.
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a,b

def main():
    #=================================Linear Regression Baseline==================================

    LinearRegression_baseline.main()#All the datasets are loaded within the function itself

    #=============================================================================================

    # =============================================================================================

    models_baseline.main()

    # =============================================================================================
    models = np.array([LinearRegression, DecisionTreeRegressor, KNeighborsRegressor, Ridge, MLPRegressor, RandomForestRegressor, ElasticNet])
    #models_names = ["LinearRegression", "DecisionTreeRegressor", "KNeighborsRegressor", "Ridge", "MLPRegressor", "RandomForestRegressor", "ElasticNet"]
    models_names = ["LR", "DTR", "KNR", "Ridge", "MLPR","RFR", "EN"]
    stock = ["ICICI","TATA","VEDL","REDDY"]
    model_count = models.shape[0]
    fig_no = np.zeros([4*model_count])
    features = SelectKBest(f_regression)
    #train_X1,train_Y1,test_X1,test_Y1 = FeatureSelection.main(train_X,train_Y,test_X,test_Y)
    rmse = []
    #k_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13]

    #*********************************************ICICI data****************************************

    for i in range (0,model_count):
        train_X, train_Y, test_X, test_Y = data.load_ICICI() #load the data

        start_time = time.clock()

        clf = models[i]()
        train_X, train_Y, test_X, test_Y, best_K = FeatureSelection.main(clf, train_X, train_Y, test_X, test_Y)
        k_value = [best_K]
        fig_no[i] = i
        hyper_parameters = hyper.main(models[i], model_count)
        hyper_parameters['Kbest__k'] = k_value
        pipeline = Pipeline([('Kbest', features), ('model', clf)])
        rand_search = GridSearchCV(pipeline, param_grid=hyper_parameters)
        rand_search.fit(train_X, train_Y)

        predictions = rand_search.predict(test_X)

        end_time = time.clock() - start_time # Calculate the speed
        filename = "..\Speed\ICICI_" + "{}".format(models_names[i]) + "_time.txt" # Write the speed to a file
        target = open(filename, 'w')
        target.write(str(end_time))

        MSE = mean_squared_error(test_Y, predictions)
        RMSE = math.sqrt(MSE)  # Calculate RMSE
        filename = "..\RMSE\ICICI_" + "{}".format(models_names[i]) + "_rmse.txt"  #Store RMSE to a file
        target = open(filename, 'w')
        target.write(str(RMSE))
        print "ICICI ",models_names[i],"  ",RMSE
        rmse.append(RMSE)
        Graphs_plotting.line_graph(fig_no[i],test_Y, predictions,models_names[i],stock[0])
    Graphs_plotting.bar_chart(101, rmse, models_names,stock[0],model_count)
    rmse = []
    # *********************************************TATA data*****************************************

    for i in range(0, model_count):
        train_X, train_Y, test_X, test_Y = data.load_TATA()

        start_time = time.clock()

        clf = models[i]()
        train_X, train_Y, test_X, test_Y, best_K = FeatureSelection.main(clf, train_X, train_Y, test_X, test_Y)
        k_value = [best_K]
        fig_no[model_count + i] = model_count + i
        pipeline = Pipeline([('Kbest', features), ('model', clf)])
        hyper_parameters = hyper.main(models[i], model_count)
        hyper_parameters['Kbest__k'] = k_value
        rand_search = GridSearchCV(pipeline, param_grid=hyper_parameters )
        rand_search.fit(train_X, train_Y)
        predictions = rand_search.predict(test_X)

        end_time = time.clock() - start_time
        filename = "..\Speed\TATA_" + "{}".format(models_names[i]) + "_time.txt"
        target = open(filename, 'w')
        target.write(str(end_time))

        MSE = mean_squared_error(test_Y, predictions)
        RMSE = math.sqrt(MSE)
        filename = "..\RMSE\TATA_" + "{}".format(models_names[i]) + "_rmse.txt"
        target = open(filename, 'w')
        target.write(str(RMSE))
        print "TATA ", models_names[i], "  ", RMSE
        rmse.append(RMSE)
        Graphs_plotting.line_graph(fig_no[model_count + i],test_Y, predictions, models_names[i],stock[1])
    Graphs_plotting.bar_chart(102, rmse, models_names,stock[1],model_count)
    rmse = []
    # *********************************************VEDL data*****************************************

    for i in range(0, model_count):
        train_X, train_Y, test_X, test_Y = data.load_VEDL()

        start_time = time.clock()

        clf = models[i]()
        train_X, train_Y, test_X, test_Y, best_K = FeatureSelection.main(clf, train_X, train_Y, test_X, test_Y)
        k_value = [best_K]
        fig_no[2*model_count + i] = 2*model_count + i
        pipeline = Pipeline([('Kbest', features), ('model', clf)])
        hyper_parameters = hyper.main(models[i], model_count)
        hyper_parameters['Kbest__k'] = k_value
        rand_search = GridSearchCV(pipeline, param_grid=hyper_parameters)
        rand_search.fit(train_X, train_Y)
        predictions = rand_search.predict(test_X)

        end_time = time.clock() - start_time
        filename = "..\Speed\VEDL_" + "{}".format(models_names[i]) + "_time.txt"
        target = open(filename, 'w')
        target.write(str(end_time))

        # if (models[i] in [MLPRegressor, KNeighborsRegressor, RandomForestRegressor]):
        #     predictions = predictions - (predictions[0] - test_Y[0])#normalizing the predicted values
        MSE = mean_squared_error(test_Y, predictions)
        RMSE = math.sqrt(MSE)
        filename = "..\RMSE\VEDL_" + "{}".format(models_names[i]) + "_rmse.txt"
        target = open(filename, 'w')
        target.write(str(RMSE))
        print "VEDANTA ", models_names[i], "  ", RMSE
        rmse.append(RMSE)
        Graphs_plotting.line_graph(fig_no[2*model_count + i], test_Y, predictions, models_names[i],stock[2])
    Graphs_plotting.bar_chart(103, rmse, models_names,stock[2],model_count)
    rmse = []
    # *********************************************REDDY data*****************************************

    for i in range(0, model_count):
        train_X, train_Y, test_X, test_Y = data.load_REDDY()

        start_time = time.clock()

        clf = models[i]()
        train_X, train_Y, test_X, test_Y, best_K = FeatureSelection.main(clf, train_X, train_Y, test_X, test_Y)
        k_value = [best_K]
        fig_no[3 * model_count + i] = 3 * model_count + i
        pipeline = Pipeline([('Kbest', features), ('model', clf)])
        hyper_parameters = hyper.main(models[i], model_count)
        hyper_parameters['Kbest__k'] = k_value
        rand_search = GridSearchCV(pipeline, param_grid=hyper_parameters)
        rand_search.fit(train_X, train_Y)
        predictions = rand_search.predict(test_X)

        end_time = time.clock() - start_time
        filename = "..\Speed\REDDY_" + "{}".format(models_names[i]) + "_time.txt"
        target = open(filename, 'w')
        target.write(str(end_time))

        MSE = mean_squared_error(test_Y, predictions)
        RMSE = math.sqrt(MSE)
        filename = "..\RMSE\REDDY_" + "{}".format(models_names[i]) + "_rmse.txt"
        target = open(filename, 'w')
        target.write(str(RMSE))
        print "DR REDDY ", models_names[i], "  ", RMSE
        rmse.append(RMSE)
        Graphs_plotting.line_graph(fig_no[3 * model_count + i], test_Y, predictions, models_names[i],stock[3])
    Graphs_plotting.bar_chart(103, rmse, models_names,stock[3],model_count)
if __name__ == "__main__": main()