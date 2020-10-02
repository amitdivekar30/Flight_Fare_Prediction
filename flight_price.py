# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 22:27:30 2020

@author: aad
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

#Importing Dataset

train_data= pd.read_excel(r"Data_Train.xlsx")

pd.set_option('display.max_columns', None)
train_data.head()

train_data.info()

train_data['Duration'].value_counts()

train_data.dropna(inplace=True)

train_data.isnull().sum()

#EDA

train_data["Journey_day"]=pd.to_datetime(train_data.Date_of_Journey, format='%d/%m/%Y').dt.day
train_data["Journey_month"]=pd.to_datetime(train_data.Date_of_Journey, format='%d/%m/%Y').dt.month
train_data.head()

train_data.drop(['Date_of_Journey'], axis=1,inplace= True)

train_data["Dep_hour"]=pd.to_datetime(train_data.Dep_Time).dt.hour
train_data["Dep_min"]=pd.to_datetime(train_data.Dep_Time).dt.minute

train_data.drop(['Dep_Time'], axis=1,inplace= True)

train_data["Arrival_hour"]=pd.to_datetime(train_data.Arrival_Time).dt.hour
train_data["Arrival_min"]=pd.to_datetime(train_data.Arrival_Time).dt.minute

train_data.drop(['Arrival_Time'], axis=1,inplace= True)

duration= list(train_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) !=2:
        if "h" in duration[i]:
            duration[i]=duration[i].strip()+ " 0m"
        else:
            duration[i]="0h "+ duration[i]
            
duration_hours = []
duration_mins = []

for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep= "h")[0]))
    duration_mins.append(int(duration[i].split(sep= "m")[0].split()[-1]))


train_data["duration_hours"] =duration_hours
train_data["duration_mins"]= duration_mins

train_data.drop(['Duration'], axis=1,inplace= True)

#handling Categorical Data
train_data["Airline"].value_counts()

sns.catplot(y='Price', x='Airline', data=train_data.sort_values("Price", ascending= False), 
            kind="boxen", height=6, aspect=3)

Airline= train_data[["Airline"]]

Airline=pd.get_dummies(Airline, drop_first=True)

Airline.head()

train_data["Source"].value_counts()

sns.catplot(y='Price', x='Source', data=train_data.sort_values("Price", ascending= False), 
            kind="boxen", height=6, aspect=3)

Source= train_data[["Source"]]

Source=pd.get_dummies(Source, drop_first=True)

Source.head()

train_data["Destination"].value_counts()

sns.catplot(y='Price', x='Destination', data=train_data.sort_values("Price", ascending= False), 
            kind="boxen", height=6, aspect=3)

Destination= train_data[["Destination"]]

Destination=pd.get_dummies(Destination, drop_first=True)

Destination.head()

train_data['Route']

train_data['Additional_Info']

train_data.drop(['Route','Additional_Info'], axis=1, inplace=True)

train_data['Total_Stops'].value_counts()

#This is ordinal categorical ,so label Encoder
train_data.replace({'non-stop':0, '1 stop':1, '2 stops':2, '3 stops':3, '4 stops':4}, inplace=True)
train_data.head()

data_train=pd.concat([train_data,Airline,Source,Destination],axis=1)

data_train.head()

data_train.drop(['Airline','Source','Destination'], axis=1,inplace=True)

data_train.shape

# Test Set
test_data= pd.read_excel(r"Test_set.xlsx")
test_data.head()
test_data.info()

test_data['Duration'].value_counts()
test_data.dropna(inplace=True)
test_data.isnull().sum()

#EDA

test_data["Journey_day"]=pd.to_datetime(test_data.Date_of_Journey, format='%d/%m/%Y').dt.day
test_data["Journey_month"]=pd.to_datetime(test_data.Date_of_Journey, format='%d/%m/%Y').dt.month
test_data.head()

test_data.drop(['Date_of_Journey'], axis=1,inplace= True)

test_data["Dep_hour"]=pd.to_datetime(test_data.Dep_Time).dt.hour
test_data["Dep_min"]=pd.to_datetime(test_data.Dep_Time).dt.minute

test_data.drop(['Dep_Time'], axis=1,inplace= True)

test_data["Arrival_hour"]=pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data["Arrival_min"]=pd.to_datetime(test_data.Arrival_Time).dt.minute

test_data.drop(['Arrival_Time'], axis=1,inplace= True)

duration= list(test_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) !=2:
        if "h" in duration[i]:
            duration[i]=duration[i].strip()+ " 0m"
        else:
            duration[i]="0h "+ duration[i]
            
duration_hours = []
duration_mins = []

for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep= "h")[0]))
    duration_mins.append(int(duration[i].split(sep= "m")[0].split()[-1]))


test_data["duration_hours"] =duration_hours
test_data["duration_mins"]= duration_mins

test_data.drop(['Duration'], axis=1,inplace= True)

#handling Categorical Data
test_data["Airline"].value_counts()

Airline= test_data[["Airline"]]

Airline=pd.get_dummies(Airline, drop_first=True)

Airline.head()

test_data["Source"].value_counts()


Source= test_data[["Source"]]

Source=pd.get_dummies(Source, drop_first=True)

Source.head()

test_data["Destination"].value_counts()


Destination= test_data[["Destination"]]

Destination=pd.get_dummies(Destination, drop_first=True)

Destination.head()

test_data['Route']

test_data['Additional_Info']

test_data.drop(['Route','Additional_Info'], axis=1, inplace=True)

test_data['Total_Stops'].value_counts()

#This is ordinal categorical ,so label Encoder
test_data.replace({'non-stop':0, '1 stop':1, '2 stops':2, '3 stops':3, '4 stops':4}, inplace=True)
test_data.head()

data_test=pd.concat([test_data,Airline,Source,Destination],axis=1)
data_test.head()
data_test.drop(['Airline','Source','Destination'], axis=1,inplace=True)
data_test.shape

##Feature Selection

data_train.shape
data_train.columns

X=data_train.loc[:, ['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min', 'duration_hours',
       'duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]

X.head()

y =data_train.iloc[:, 1]

#find correlation betweeen Independent and Dependent attributes
plt.figure(figsize=(18,18))
sns.heatmap(train_data.corr(), annot=True, cmap="RdYlGn")
plt.show()

# Important featureusing ExtraTreesRegressor

from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X,y)

#plot graph of featureimporatances for better visualization
plt.figure(figsize=(12,8))
feat_importances= pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()

# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
reg_rf =RandomForestRegressor()
reg_rf.fit(X_train, y_train)

y_pred= reg_rf.predict(X_test)
reg_rf.score(X_train, y_train)
reg_rf.score(X_test, y_test)

sns.distplot(y_test-y_pred)

plt.scatter(y_test,y_pred,alpha=0.5)
plt.xlabel("y_test")
plt.ylabl("y_pred")
plt.show()

from sklearn import metrics
print("MAE: ", metrics.mean_absolute_error(y_test, y_pred))
print("MSE: ", metrics.mean_squared_error(y_test, y_pred))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("R2 square: ", metrics.r2_score(y_test, y_pred))


# Hyperparameter Tuning
# Choose following method for hyperparameter tuning
# RandomizedSearchCV --> Fast
# GridSearchCV
# Assign hyperparameters in form of dictionery
# Fit the model
# Check best paramters and best score

from sklearn.model_selection import RandomizedSearchCV


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

rf_random.fit(X_train,y_train)
rf_random.best_params_

prediction = rf_random.predict(X_test)

plt.figure(figsize = (8,8))
sns.distplot(y_test-prediction)
plt.show()

plt.figure(figsize = (8,8))
plt.scatter(y_test, prediction, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()

print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


import pickle
# open a file, where you ant to store the data
file = open('flight_rf.pkl', 'wb')

# dump information to that file
pickle.dump(reg_rf, file)


model = open('flight_rf.pkl','rb')
forest = pickle.load(model)

y_prediction = forest.predict(X_test)

metrics.r2_score(y_test, y_prediction)


