#Business problem : Whether the client has subcribed a term deposit or not in a bank



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

data = pd.read_csv("/home/shiva/Desktop/Pipelines/data/train.csv")

#EDA

data.shape
#(45211, 32)

#the data contains 32 features/variables
# 1. Feature Engineering
# By performing feature Engineering, we will be knowing:
# what features to be considered with respective to the target variable 

data.describe()
#By looking the data, data to be scaled.

data.columns

#'age', 'default', 'balance', 'housing', 'loan', 'duration', 'campaign','pdays', 'previous', 'poutfailure', 'poutother', 'poutsuccess',
#'poutunknown', 'con_cellular', 'con_telephone', 'con_unknown','divorced', 'married', 'single', 'joadmin.', 'joblue.collar',
#'joentrepreneur', 'johousemaid', 'jomanagement', 'joretired','joself.employed', 'joservices', 'jostudent', 'jotechnician', jounemployed', 'jounknown', 'y'

# Target variable named "y" 
# Binary target variable 0 or 1
data["y"].unique()

# based on the age, balance, loan, job, marital statuts we need to predict the term deposit is done or not

data["y"].value_counts()
#Imbalanced data
#To overcome this, three techniques can be used
# 1. Oversampling - Replicating the minority class
# 2. Undersampling - Reducing data of the majority class
# 3. SMOTE(Synthetic Minority Oversampling TEchnique)
#        It is a synthetic minority oversampling technique, which makes synthetic data points by finding the nearest neighbours to each minority sample.


#Find any duplicates in the data
data.duplicated()

