# -*- coding: utf-8 -*-
"""Group9_SportsPrediction (1).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kB5fd7Tkfo1u7uXVuQh0ZblaKxD-DeXq
"""

#importing all important libraries
import pandas as pd
import os
import sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
import numpy as np, pandas as pd
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold,GridSearchCV
from google.colab import drive
drive.mount('/content/drive')

#fifa data set to be used
training_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/players_21.csv')
testing_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/players_22.csv')

"""## **1. Data Preparation and Feature Extraction**

Training Data preprocessing
"""

# These are columns that contain ids and url links.
training_data.drop('sofifa_id', axis = 1, inplace = True)
training_data.drop('player_url', axis = 1, inplace = True)
training_data.drop('player_face_url', axis = 1, inplace = True)
training_data.drop('club_logo_url', axis = 1, inplace = True)
training_data.drop('club_flag_url', axis = 1, inplace = True)
training_data.drop('nation_logo_url', axis = 1, inplace = True)
training_data.drop('nation_flag_url', axis = 1, inplace = True)
training_data.drop('nationality_id', axis = 1, inplace = True)
training_data.drop('nation_team_id', axis = 1, inplace = True)
training_data.drop('club_team_id', axis = 1, inplace = True)
training_data.drop('dob', axis = 1, inplace = True) #dropped dob because it basically does the same work as age

# drop columns with 30% or more missing data
missing = (training_data.isnull().sum() / len(training_data)) * 100
columns_missing = missing[missing >= 30].index
training_data.drop(columns=columns_missing, inplace =True)

training_data.info()

Y = pd.DataFrame()
Y = training_data['overall'] # separate the overall column as the Y
training_data.drop('overall', axis = 1, inplace = True) # drop overall from the training data
Y

#grouped columns that are numeric
numeric_training = training_data.select_dtypes(include=['int', 'float']).columns # extract numeric data
numeric_training = training_data[numeric_training]
numeric_training.info()

#impute the numeric data using median
imp = SimpleImputer(strategy = 'median')
numeric_training = pd.DataFrame(imp.fit_transform(numeric_training), columns=numeric_training.columns)
numeric_training.info()

#grouped all columns with non-numeric values and are objects

object_training = training_data.select_dtypes(exclude= ['int', 'float']).columns # extract the object data
object_training = training_data[object_training]
object_training.info()

#imputed the non numeric values using most_frequent ie. to fill in with the most appearing
imp = SimpleImputer(strategy='most_frequent') # impute the object data
object_training = pd.DataFrame(imp.fit_transform(object_training), columns = object_training.columns)
object_training.info()

#encode the object data using the label encoder
encoded_object = pd.DataFrame() # encode the object data
for c in object_training.columns:
  label_encoder = LabelEncoder()
  encoded_object[c] = label_encoder.fit_transform(object_training[c])
encoded_object.info()

# combine numeric column and object column with overall
training_data = pd.DataFrame()
training_data = pd.concat([encoded_object, numeric_training], axis = 1)
training_data['overall'] = Y
training_data.info()

"""# **2. Feature Subsets**"""

#removing overall from the training data
Y = training_data['overall'] # extract overall
training_data.drop('overall', axis = 1, inplace = True) # drop overall
training_data.info()

#Training using rf for Feature Importance
model=RandomForestRegressor()
model.fit(training_data,Y)

#Getting the Important Features from our dataset
name_of_feature=training_data.columns #contains the names of features in our dataset
feature_importance=model.feature_importances_ #contains the scores of each feature

#Sorting the Feature Importance
feature_importance_df=pd.DataFrame({'Feature':name_of_feature,'Importance':feature_importance}) #creates a dataframe with the names of our feature and corresponding scores
feature_importance_df=feature_importance_df.sort_values(by='Importance',ascending=False) #sorts our results from highest to lowest
feature_importance_df

#Using the first 10 features
first_10_features = feature_importance_df['Feature'].values[:11]
first_10_features

#updates to our new training_x value
training_data=training_data[first_10_features]

#scaling the data
sc = StandardScaler()  # scale the data set to be used in the model
training_x = sc.fit_transform(training_data)
scaler2021=sc
training_x = pd.DataFrame(training_x, columns = training_data.columns)
training_x.head(5)

"""# **3. Machine learning model with cross validation**"""

Xtrain,Xtest,Ytrain,Ytest=train_test_split(training_x,Y,test_size=0.2,random_state=42) #splitting our data

dt = DecisionTreeRegressor(criterion ='squared_error', random_state = 42)
gb = GradientBoostingRegressor(init = dt, n_estimators = 200, learning_rate = 0.01)
XGB_model = xgb.XGBRegressor(objective="reg:squarederror")
rf=RandomForestRegressor(n_estimators=1000, n_jobs = -1)

# Gradient Boosting Regressor
gb.fit(Xtrain,Ytrain)

# Cross validation with the gradient boosting regresssor
cv=KFold(n_splits=5)
PARAMETERS ={
"max_depth":[5,10, 12],
"n_estimators":[50,70,100],
"max_features":['sqrt', 'log2'],
"criterion":['squared_error', 'friedman_mse'],
"random_state": [20, 22, 24]}
gb_cv = GridSearchCV(estimator = gb,param_grid=PARAMETERS,cv=cv,scoring="neg_mean_absolute_error")
gb_cv.fit(Xtrain, Ytrain)

best_parameters = gb_cv.best_params_
print("Best parameters:", best_parameters) # display the best parameters for the gradient boosting regressor

y_pred=gb_cv.predict(Xtest)
print("The mean absolute error is ",(mean_absolute_error(y_pred,Ytest))) # display the mean absolute error

# Random Forest Regressor
rf.fit(Xtrain, Ytrain)

# Cross validation with the random forest regresssor
cv=KFold(n_splits=5)
PARAMETERS ={
"max_depth": [10, 12, 15],
"n_estimators": [50, 70, 100],
"criterion": ['squared_error', 'friedman_mse'],
"max_features": ['sqrt', 'log2'],
"random_state": [20, 22, 24]}
rf_cv = GridSearchCV(estimator = rf,param_grid=PARAMETERS,cv=cv,scoring="neg_mean_absolute_error")
rf_cv.fit(Xtrain, Ytrain)

best_parameters = rf_cv.best_params_
print("Best parameters:", best_parameters) # display the best parameters for the random forest regressor

y_pred=rf_cv.predict(Xtest)
print("The mean absolute error is ",(mean_absolute_error(y_pred,Ytest))) # display the mean absolute error

# Cross validation with the XGB regresssor
cv=KFold(n_splits=5)
PARAMETERS ={
"max_depth": [10, 12, 15],
"random_state": [20, 22, 24]}
xgb_cv = GridSearchCV(estimator = XGB_model,param_grid=PARAMETERS,cv=cv,scoring="neg_mean_absolute_error")
xgb_cv.fit(Xtrain, Ytrain)

best_parameters = xgb_cv.best_params_
print("Best parameters:", best_parameters) # display the best parameters for the XGB regressor

y_pred=xgb_cv.predict(Xtest)
print("The mean absolute error is ",(mean_absolute_error(y_pred,Ytest))) # display the mean absolute error

"""# **4. Model Optimization**
Fine tuning , training and re-testing


"""

training_x.info()

Xtrain,Xtest,Ytrain,Ytest=train_test_split(training_x,Y,test_size=0.2,random_state=42) #splitting our data

#cross validation with gb boost which was our best performing model
cv=KFold(n_splits=5)
PARAMETERS ={
"max_depth":[7,10,15],
"n_estimators":[50,70,120],
"max_features":['sqrt', 'log2'],
"criterion":['squared_error', 'friedman_mse'],
"random_state": [20, 22, 24]}
gb_cv = GridSearchCV(estimator = gb,param_grid=PARAMETERS,cv=cv,scoring="neg_mean_absolute_error")
gb_cv.fit(Xtrain, Ytrain)

#display the best parameters for the GB regressor
best_parameters = gb_cv.best_params_
print("Best parameters:", best_parameters)

y_pred=gb_cv.predict(Xtest)
print("The mean absolute error is ",(mean_absolute_error(y_pred,Ytest))) # display the mean absolute error

"""# **5. Model testing with player 22 data**"""

#using specific columns used in our training data in our testing data
new_testing_data = testing_data[training_x.columns]
new_testing_data.info()

Y = testing_data['overall']
Y

#grouped columns that are numeric
numeric_testing = new_testing_data.select_dtypes(include=['int', 'float']).columns
numeric_testing = new_testing_data[numeric_testing]
numeric_testing.info()

#grouped all columns with non-numeric values and object
object_testing = new_testing_data.select_dtypes(exclude=['int', 'float']).columns
object_testing = new_testing_data[object_testing]
object_testing.info()

#imputed the numeric values using most_frequent ie. to fill in with the mean
imp = SimpleImputer(strategy='mean')
numeric_testing = pd.DataFrame(imp.fit_transform(numeric_testing), columns = numeric_testing.columns)
numeric_testing.info()

#encode the object data using the label encoder
encoded_object = pd.DataFrame()
for c in object_testing.columns:
  label_encoder = LabelEncoder()
  encoded_object[c] = label_encoder.fit_transform(object_testing[c])
encoded_object.info()

#rejoined the numeric and non-numeric columns
testing_x = pd.DataFrame()
testing_x = pd.concat([encoded_object, numeric_testing], axis = 1)
testing_x.info()

sc = StandardScaler()
new_testing_x = sc.fit_transform(testing_x)
new_testing_x = pd.DataFrame(new_testing_x, columns = testing_x.columns)
new_testing_x.head(5)

Xtrain,Xtest,Ytrain,Ytest=train_test_split(testing_x,Y,test_size=0.2,random_state=42)#split data

gb = GradientBoostingRegressor(n_estimators = 50, random_state = 20) #using the best perfomind model to train our 2022 data

gb.fit(Xtrain, Ytrain)

y_pred = gb.predict(Xtest)
print("The mean absolute error is ",(mean_absolute_error(y_pred,Ytest))) # display the mean absolute error

"""# 6. Saving the model using pickle"""

import pickle

# deployment_model = '/content/drive/MyDrive/Colab Notebooks/deployment_model.pkl'
# pickle.dump(gb, open(deployment_model, 'wb'))
# loaded_model = pickle.load(open(deployment_model, 'rb'))

with open("deployment_model.pkl", 'wb') as file:
  pickle.dump((gb,scaler2021), file)

# scaler_model = '/content/drive/MyDrive/Colab Notebooks/scalar_model.pkl'
# pickle.dump(sc,open(scaler_model, 'wb'))
# loaded_model = pickle.load(open(scaler_model, 'rb'))

with open("scalar_model.pkl", 'wb') as file:
  pickle.dump((sc, file))