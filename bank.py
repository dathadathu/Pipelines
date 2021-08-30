#Business problem : Whether the client has subcribed a term deposit or not in a bank 
#Market targeting

#based on the age, balance, loan, job, marital statuts we need to predict the term deposit is done or not



#Importing necessary packages

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

#Load the data

data = pd.read_csv("/home/shiva/Desktop/Pipelines/data/train.csv",delimiter=";")

data.head()
#EDA

data.shape
#(45211, 17)
#the data contains 17 features/variables

data.describe()
#By looking the data, data conversion to be done
#label encoding - categorical 
#one hot encoding - binary

data.columns

#'age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
#'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
#'previous', 'poutcome', 'y'

# Target variable named "y" 
# Binary target variable 0 or 1
data["y"].unique()

data["y"].value_counts()
#Imbalanced data
#To overcome this, three techniques can be used
# 1. Oversampling - Replicating the minority class
# 2. Undersampling - Reducing data of the majority class
# 3. SMOTE(Synthetic Minority Oversampling TEchnique)
#        It is a synthetic minority oversampling technique, which makes synthetic data points by finding the nearest neighbours to each minority sample.


#Find any duplicates in the data
data.duplicated().sum()

#there are no duplicate values in the data

#find any null values in the data
data.isnull().sum()

#No null values

#Check the data types of the data

data.info()
#int and object type are there

for col in data.columns:
    if data[col].dtype == "object":
        print("*"*40)
        print("\n", data[col].value_counts())
    else:
        pass

#we can observe the output lot of unknown values
#In Job, Education, Contact, Poutcome



