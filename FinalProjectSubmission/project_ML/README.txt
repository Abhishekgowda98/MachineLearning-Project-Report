Authors: Lopamudra Pal, Sanketh Kokkodu Balakrishna

Files and Descriptions:
----------------------
Pre-processing: Stock_preprocessing.py. Needs data sets and path to be set. Preprocesses and writes to a .csv file.
Downloaded Data in folder DataSets
Processed Data in folder Data
Code:- 
All code is in folder Code_Stock Prediction
Executing models.py calls all the other files.
	data.py - python file containing function to load the data
	models_baseline.py - python file containing all models for all datasets without feature selection or hyperparameter selection
	LinearRegression_baseline.py - python file containing LinearRegression model (default, no hyperparameters) as the baseline for different stocks
	Graphs_plotting.py - python file containing function for plotting line charts and bar charts
	hyper.py - python file containing function to create a dictionary of hyperparameter values to be given as input to sklearn GridSearchCV
	BestModel.py - python file containing the best model for ICICI
	models.py - python file containing all Regression models along with feature selection and hyperparameter selection 
	FeatureSelection.py - python file which selects the best features using sklearn's SelectKBest method
	
Figures:-
	All figures in folder called 'Figures'
	.jpeg figures are Bar Graphs plotting the RMSE error rate for different regression models
	
	

	
Speed folder contains the prediction times of different models and different stocks
RMSE folder contains the prediction error of different models and different stocks
Time_&_RMSE.xlsx - Microsoft Excel file containing Speed and RMSE values of different models for different stocks. These models are the ones which include FeatureSelection and HyperparameterSelection