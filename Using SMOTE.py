#%%
from time import time
import pandas as pd
import numpy as np
from scipy.sparse.construct import random
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

import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE

#%%
#%%
data = pd.read_csv("/home/shiva/Desktop/Pipelines/data/train.csv",delimiter=";")

#%%
data.head()
#%%
#EDA

data.shape
#(45211, 17)
#the data contains 17 features/variables
#%%
data.describe()
#By looking the data, data conversion to be done
#label encoding - categorical 
#one hot encoding - binary
#%%
data.columns

#'age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
#'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
#'previous', 'poutcome', 'y'
#%%
# Target variable named "y" 
# Binary target variable 0 or 1
data["y"].unique()

#%%
data["y"].value_counts()
#Imbalanced data
#To overcome this, three techniques can be used
# 1. Oversampling - Replicating the minority class
# 2. Undersampling - Reducing data of the majority class
# 3. SMOTE(Synthetic Minority Oversampling TEchnique)
#        It is a synthetic minority oversampling technique, which makes synthetic data points by finding the nearest neighbours to each minority sample.

#%%
#Find any duplicates in the data
data.duplicated().sum()

#there are no duplicate values in the data
#%%
#find any null values in the data
data.isnull().sum()

#No null values
#%%
#Check the data types of the data

data.info()
#int and object type are there

#%%
#Heatmap of the data
sns.heatmap(data.corr())


#%%
for col in data.columns:
    if data[col].dtype == "object":
        print("*"*40)
        print("\n", data[col].value_counts())
    else:
        pass

#we can observe the output lot of unknown values
#In Job, Education, Contact, Poutcome

#%%
#We can see there is space before the value 
#Using strip trying to remove 
#Defining function for this

# Creating a function which will remove extra leading 
# and tailing whitespace from the data.
# pass dataframe as a parameter here

#%%
def whitespace_remover(dataframe):
    
    # iterating over the columns
    for i in dataframe.columns:
          
        # checking datatype of each columns
        if dataframe[i].dtype == 'object':
              
            # applying strip function on column
            dataframe[i] = dataframe[i].map(str.strip)
        else:
              
            # if condn. is False then it will do nothing.
            pass


#%%
# applying whitespace_remover function on dataframe
whitespace_remover(data)

#%%
data["y"] = data["y"].map({'yes':1,'no':0})

#%%
#as there are categorical values
#transforming using one hot encoding
#In pandas, we have pd.get_dummies can mention the columns

#before we need to seprate independent and dependent columns

y = data["y"] 
X = data.drop("y",axis = 1)

# %%
X
# %%
y
#%%

####%%time
cat_cols = []
for i in data.columns:
    if data[i].dtype == "object":
        #data[data[i] == "Unknown"]
        print(i)
        cat_cols.append(i)
    else:
        pass

#%%
cat_cols

#%%

X_d = pd.get_dummies(X,drop_first=False,columns=cat_cols)
#%%
#del X_train,X_test,y_test,y_train

#%%
X_train,X_test,y_train,y_test = train_test_split(X_d,y,test_size=0.3,random_state=42)

# %%
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

# %%
print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

#%%
sm = SMOTE(random_state=2)
#%%
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())

#%%
print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
# %%
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report

#%%
parameters = {
    'C': np.linspace(1, 10, 10)
             }

#%%             

parameters.values()


#%%
sv = RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)
clf = GridSearchCV(sv, parameters, cv=5, verbose=5, n_jobs=-1)

#%%
clf.fit(X_train_res, y_train_res.ravel())
# %%
clf.best_params_

#%%
svvv = SVC(C=1,kernel='sigmoid',random_state=42)

#%%
svvv.fit(X_train_res, y_train_res.ravel())
# %%
import itertools

#%%
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# %%
y_train_pre = svvv.predict(X_train)

cnf_matrix_tra = confusion_matrix(y_train, y_train_pre)

print("Recall metric in the train dataset: {}%".format(100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])))


class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Confusion matrix')
plt.show()
# %%
y_pre = svvv.predict(X_test)

cnf_matrix = confusion_matrix(y_test, y_pre)

print("Recall metric in the testing dataset: {}%".format(100*cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])))
#print("Precision metric in the testing dataset: {}%".format(100*cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[1,0])))
# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix , classes=class_names, title='Confusion matrix')
plt.show()
# %%
