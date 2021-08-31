#Business problem : Whether the client has subcribed a term deposit or not in a bank 
#Market targeting

#based on the age, balance, loan, job, marital statuts we need to predict the term deposit is done or not



#Importing necessary packages

#%%
from time import time
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

import matplotlib.pyplot as plt
import seaborn as sns

#Load the data
#As i observed space before values in some columns 
#To overcome this, "skipinitalspace = True" while loading the data
#%%
data = pd.read_csv("/home/shiva/Desktop/Pipelines/data/train.csv",delimiter=";",)

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
#Getting unique values of the columns

for c in data.columns:
    print("*"*40)
    print("Name of the Column: ", c)
    print(data[c].unique())

# %%
def distribution(dataframe):

    for cols in dataframe.columns:
        if dataframe[cols].dtype in ["int64","float64"]:
            plt.hist(dataframe[cols],label=True)
            plt.title("Histogram of {}".format(cols))
            plt.show()
        
        else:
            sns.countplot(x = cols, data = dataframe)
            plt.title("Count of {}".format(cols))
            plt.show()
# %%
%%time
distribution(data)


# %%
%%time
cat_cols = []
for i in data.columns:
    if data[i].dtype == "object":
        #data[data[i] == "Unknown"]
        print(i)
        cat_cols.append(i)
    else:
        pass


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
cat_cols.pop()

#%%

X_d = pd.get_dummies(X,drop_first=False,columns=cat_cols)
#%%
del X_train,X_test,y_test,y_train

#%%
X_train,X_test,y_train,y_test = train_test_split(X_d,y,test_size=0.3,random_state=42)
# %%
pipeline_lr = Pipeline([
    ('ss', StandardScaler()),
    ('lr', LogisticRegression())
    ])

# %%
%%time
pipeline_lr.fit(X_train,y_train)

# %%
%%time
y_predict = pipeline_lr.predict(X_test)
print('Test Accuracy Score: {:.4f}'.format(accuracy_score(y_test, y_predict)))
# %%

pipeline_lr_mm = Pipeline([
    ('mms', MinMaxScaler()),
    ('lr', LogisticRegression())
    ])
    
pipeline_lr_r = Pipeline([
    ('rs', RobustScaler()),
    ('lr', LogisticRegression())
    ])

pipeline_lr_w = Pipeline([
    ('lr', LogisticRegression())
    ])

pipeline_lr_s = Pipeline([
    ('ss', StandardScaler()),
    ('lr', LogisticRegression())
    ])
# %%
pipeline_dict = {
    0: 'Logistic Regression without scaler',
    1: 'Logistic Regression with MinMaxScaler',
    2: 'Logistic Regression with RobustScaler',
    3: 'Logistic Regression with StandardScaler',
    }
pipeline_dict
# %%
pipelines = [pipeline_lr_w, pipeline_lr_mm, pipeline_lr_r, pipeline_lr_s]
# %%
for p in pipelines:
    p.fit(X_train, y_train)
# %%
for i, val in enumerate(pipelines):
    print('%s pipeline Test Accuracy Score: %.4f' % (pipeline_dict[i], accuracy_score(y_test, val.predict(X_test))))
# %%
%%time
pipeline_knn = Pipeline([
    ('ss1', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=4))
    ])
pipeline_dt = Pipeline([
    ('ss2', StandardScaler()),
    ('dt', DecisionTreeClassifier())
    ])
pipeline_rf = Pipeline([
    ('ss3', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=80))
    ])
pipeline_lr = Pipeline([
    ('ss4', StandardScaler()),
    ('lr', LogisticRegression())
    ])
pipeline_svm_lin = Pipeline([
    ('ss5', StandardScaler()),
    ('svm_lin', SVC(kernel='linear'))
    ])
pipeline_svm_sig = Pipeline([
    ('ss6', StandardScaler()),
    ('svm_sig', SVC(kernel='sigmoid'))
    ])
# %%
%%time
pipeline_dicti = {
    0: 'knn',
    1: 'dt',
    2: 'rf',
    3: 'lr',
    4: 'svm_lin',
    5: 'svm_sig',
    
    }
pipeline_dicti
# %%
%%time
pipelines = [pipeline_lr, pipeline_svm_lin, pipeline_svm_sig, pipeline_knn, pipeline_dt, pipeline_rf]

# %%
%%time
for p in pipelines:
    p.fit(X_train, y_train)
# %%
%%time
l = []
for i, val in enumerate(pipelines):
    l.append(accuracy_score(y_test, val.predict(X_test)))

#%% 
%%time   
result_df = pd.DataFrame(list(pipeline_dicti.items()),columns = ['Idx','Estimator'])

#%%
result_df["Test_Accuracy"] = l

# %%
result_df

#%%
from sklearn.metrics import confusion_matrix

#%%
model = SVC(kernel='sigmoid',random_state=42)

#%%
model.fit(X_train,y_train)

#%%
y_pred = model.predict(X_test)

#%%
acc = accuracy_score(y_test,y_pred)

#%%
print(acc)

#%%





# %%

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
#%%
scores = ['precision', 'recall']
#%%
%%time

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},
                    {'kernel': ['sigmoid'], 'C': [1, 10, 100, 1000]}]


#%%
%%time
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
# %%
