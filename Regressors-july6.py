#!/usr/bin/env python
# coding: utf-8

# # Importing **Libraries**

# In[1]:


get_ipython().system('pip install eli5')


# In[2]:


# Time 
import time
import datetime
from datetime import time

# To alert warning
import warnings
warnings.filterwarnings('ignore')

# Linear algebra
import numpy as np

# Statistical functions
import scipy.stats as st

# Data processing
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm_notebook

# nltk
import nltk
import string
from numpy import array
from textblob import TextBlob
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

# Data visualization
import seaborn as sns
import plotly.offline as py
from matplotlib import style
from matplotlib import pyplot
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
#from scikitplot.metrics import plot_confusion_matrix,plot_precision_recall_curve

# To plot your graphs offline inside a Jupyter Notebook Environment
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# Sklearn libraries
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfTransformer

# eli5 to order the features based on weights
import eli5
from eli5.sklearn import PermutationImportance


# 
# # Importing Regression Modules
# 

# In[3]:


# Importing Classifier Modules
import json
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


#Algorithms
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import linear_model, preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

# Parameter Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV


# # Load & Explore Dataset
# 
# 
# 

# In[4]:


# Read the data
df_train = pd.read_csv('trainAllfeatures.csv')
df_dev = pd.read_csv('testAllfeatures.csv')


# In[5]:


# Dataset is now stored in a Pandas Dataframe
# Print the Train Data
df_train.head()


# In[6]:


# Print the Dev Data
df_dev.head()


# In[ ]:


# Print the Train Info
df_train.info()


# In[7]:


# Print the Dev Info
df_dev.info()


# In[8]:


#Size Dataset
df_train.shape , df_dev.shape


# In[9]:


#count of meangrade in the train data
df_train.groupby('Event').count()


# In[10]:


#data type for train data
pd.DataFrame(df_train.dtypes).transpose()


# In[11]:


#data type for dev data
pd.DataFrame(df_dev.dtypes).transpose()


# # Check for Missing data & data type

# In[12]:


#Finding the missing values in the train and dev data
train_missing = df_train.isnull().sum().sum()
dev_missing = df_dev.isnull().sum().sum()
print('Missing values in the train data :',train_missing)
print('Missing values in the dev data :',dev_missing)
#df_train.isnull().sum()#4
#df_dev.isnull().sum()#2


# # Describe Data
# 

# In[13]:


# describe train
df_train.describe()


# In[14]:


# describe dev
df_dev.describe()


# # Scaling my Train Data

# In[15]:


# To transform the train data between same range
x_features_train = df_train.drop(['PatientID', 'target'], axis=1)
#y_label = df_train['Forecast _Id']

#scaling dataset
sc = StandardScaler()
sc.fit(x_features_train)
X_scaled_train = sc.transform(x_features_train)
X_scaled_train = pd.DataFrame(X_scaled_train, columns=x_features_train.columns)
X_scaled_train.head()


# # Scaling my Dev Data

# In[16]:


# To transform the dev data between same range
x_features_dev = df_dev.drop(['PatientID',], axis=1)

#scaling dataset
sc = StandardScaler()
sc.fit(x_features_dev)
X_scaled_dev = sc.transform(x_features_dev)
X_scaled_dev = pd.DataFrame(X_scaled_dev, columns=x_features_dev.columns)
X_scaled_dev.head()


# # Feature Engineering

# In[17]:


#Feature engineering
#Permutation variable importance measure in a random forest for classification
x= df_train.drop(columns=['PatientID'],axis=1)
y_label = df_train['target']
lab_enc = preprocessing.LabelEncoder()
encoded_meanGrade = lab_enc.fit_transform(y_label)


# In[18]:


#build simple model to find features which are more important
#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
rfreg_model=RandomForestRegressor(n_estimators=100,random_state=42)
#fitting the model
rfreg_model.fit(X_scaled_train,y_label)


# In[19]:


# Calculate weights and show important features using eli5 library
# Permutation importance
import eli5
from eli5.sklearn import PermutationImportance
perm_imp=PermutationImportance(rfreg_model,random_state=42)
#fitting the model
perm_imp.fit(X_scaled_train,y_label)


# # Important Features

# In[20]:


#Important features for train data
eli5.show_weights(perm_imp,feature_names=X_scaled_train.columns.tolist(),top=100)


# In[21]:


#Important features for dev data
#No need for this step
#eli5.show_weights(perm_imp,feature_names=X_scaled_dev.columns.tolist(),top=100)


# # Feature Selection

# In[22]:


from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
rf = RandomForestRegressor( max_depth=5, verbose=3, n_estimators=100)
parameters = {'n_estimators': [120, 100], 'max_depth':[3,5,]}
grid = GridSearchCV(rf, parameters, cv=3, n_jobs=-1, verbose=3, scoring=make_scorer(mean_squared_error))


# In[23]:


feat_labels=['PatientID', 'age', 'target','Event',  'Mstage',	'Nstage',	'Tstage',	'original_shape_Compactness1',	'original_shape_Compactness2',	'original_shape_Maximum3DDiameter',	'original_shape_SphericalDisproportion',	'original_shape_Sphericity',	'original_shape_SurfaceArea',	'original_shape_SurfaceVolumeRatio',	'original_shape_VoxelVolume',	'original_firstorder_Energy',	'original_firstorder_Entropy',	'original_firstorder_Kurtosis',	'original_firstorder_Maximum',	'original_firstorder_Mean',	'original_firstorder_MeanAbsoluteDeviation',	'original_firstorder_Median',	'original_firstorder_Minimum',	'original_firstorder_Range',	'original_firstorder_RootMeanSquared',	'original_firstorder_Skewness',	'original_firstorder_StandardDeviation',	'original_firstorder_Uniformity',	'original_firstorder_Variance',	'original_glcm_Autocorrelation',	'original_glcm_ClusterProminence',	'original_glcm_ClusterShade',	'original_glcm_ClusterTendency',	'original_glcm_Contrast',	'original_glcm_Correlation',	'original_glcm_DifferenceEntropy',	'original_glcm_DifferenceAverage',	'original_glcm_JointEnergy',	'original_glcm_JointEntropy',	'original_glcm_Id',	'original_glcm_Idm',	'original_glcm_Imc1',	'original_glcm_Imc2',	'original_glcm_Idmn',	'original_glcm_Idn',	'original_glcm_InverseVariance',	'original_glcm_MaximumProbability',	'original_glcm_SumAverage',	'original_glcm_SumEntropy',	'original_glrlm_ShortRunEmphasis',	'original_glrlm_LongRunEmphasis',	'original_glrlm_GrayLevelNonUniformity',	'original_glrlm_RunLengthNonUniformity',	'original_glrlm_RunPercentage',	'original_glrlm_LowGrayLevelRunEmphasis',	'original_glrlm_HighGrayLevelRunEmphasis',	'original_glrlm_ShortRunLowGrayLevelEmphasis',	'original_glrlm_ShortRunHighGrayLevelEmphasis',	'original_glrlm_LongRunLowGrayLevelEmphasis',	'original_glrlm_LongRunHighGrayLevelEmphasis']


# In[24]:


# Ranking Features for train data
rnd_clf_reg = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rnd_clf_reg.fit(X_scaled_train, y_label)
importances = pd.DataFrame({'feature': X_scaled_train.columns, 'importance': np.round(rnd_clf_reg.feature_importances_, 3)})
importances = importances.sort_values('importance', ascending=False).set_index('feature')
print("Rank Features :")
#print(importances)
importances.head(22)


# In[25]:


#plot features importances
importances.plot.bar()


# In[65]:


features= df_train.drop(columns=['PatientID'],axis=1)


# In[66]:


# The importance for each feature
for features in zip(feat_labels, rnd_clf_reg.feature_importances_):
    print (features)
    


# In[ ]:


#calculate score
#بدي افصل الfeatures
#good from bad
#


# In[28]:


# print the features that greater than 0.005
#features>0.005
cc=0
for features in zip(feat_labels, rnd_clf_reg.feature_importances_):
    if features[1]>=0.005:
        print(features[0], "", end='') #124
        cc+=1
print('\n')        
print(cc)


# In[33]:


#features<0.005
cc=0
for features in zip(feat_labels, rnd_clf_reg.feature_importances_):
    if features[1]<0.005:
        print(features[0], "", end='') #29
        cc+=1
print('\n')        
print(cc)


# In[34]:


drop=['Mstage', 'Tstage', 'original_firstorder_Entropy', 'original_firstorder_Mean', 'original_firstorder_StandardDeviation', 'original_firstorder_Variance', 'original_glcm_Autocorrelation', 'original_glcm_DifferenceEntropy', 'original_glcm_Id', 'original_glcm_Idm', 'original_glrlm_RunPercentage']


# In[35]:


# Drop Features
features_drop = ['Mstage', 'Tstage', 'original_firstorder_Entropy', 'original_firstorder_Mean', 'original_firstorder_StandardDeviation', 'original_firstorder_Variance', 'original_glcm_Autocorrelation', 'original_glcm_DifferenceEntropy', 'original_glcm_Id', 'original_glcm_Idm', 'original_glrlm_RunPercentage']
train_drop_features = df_train.drop(features_drop, axis=1)
dev_drop_features = df_dev.drop(features_drop, axis=1)


# In[36]:


train_drop_features


# In[37]:


dev_drop_features


# In[46]:


train_features = train_drop_features.drop(['target','Event'], axis=1)
dev_features = dev_drop_features.drop([], axis=1)


# ### **XGB classifier**

# In[53]:


#Baseline #XGB
import datetime, time, json
import xgboost as xgb

print("Starting training at", datetime.datetime.now())
t0 = time.time()
xgb_reg_model = xgb.XGBRegressor(objective="reg:linear", random_state=42)

#model = XGBClassifier()
xgb_reg_model.fit(train_features,y_label)
y_pred_xgb_reg = xgb_reg_model.predict(dev_features)
for i in range(125):
    print(y_pred_xgb_reg[i])
    
t1 = time.time()
print("Training ended at", datetime.datetime.now())
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))


# In[60]:


# prepare submission dataframe
sub = pd.DataFrame({'PatientID':df_dev['PatientID'], 'SurvivalTime':y_pred_xgb_reg})

# write predictions to a CSV file
sub.to_csv("xgboostsub.csv", index=False)


# # Linear SVM

# In[54]:


# linear SVM in regression
import datetime, time, json
from sklearn.svm import SVR
print("Starting training at", datetime.datetime.now())
t0 = time.time()

#linear_svc = LinearSVC(dual=False)
regressor_SVM = SVR(kernel='rbf')
regressor_SVM.fit(train_features,y_label)
y_pred_svm_reg = regressor_SVM.predict(dev_features)
for i in range (125):
    print(y_pred_svm_reg[i])


t1 = time.time()
print("Training ended at", datetime.datetime.now())
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))


# # Random Forest

# In[41]:


#RandomForest_grid search
from sklearn.ensemble import RandomForestRegressor
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8]
}

random_forest_reg = RandomForestRegressor()
GS = GridSearchCV(random_forest_reg, param_grid, verbose=1, return_train_score=True)
GS.fit(train_features, y_label)
print("Best Parameters : \n", GS.best_params_)


# In[55]:


import datetime, time, json
print("Starting training at", datetime.datetime.now())
t0 = time.time()

random_forest_reg = RandomForestRegressor(max_features='auto', n_estimators=200, max_depth=8)
random_forest_reg.fit(train_features, y_label)
Y_pred_RF_reg = random_forest_reg.predict(dev_features)

for i in range(125):
    print(Y_pred_RF_reg[i])
t1 = time.time()
print("Training ended at", datetime.datetime.now())
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))


# # Linear Regression

# In[ ]:


import datetime, time, json
print("Starting training at", datetime.datetime.now())
t0 = time.time()
lin_reg = LinearRegression()
lin_reg.fit(train_features, y_label)
Y_pred_lin_reg =lin_reg.predict(dev_features)

for i in range(2419):
  print(Y_pred_lin_reg[i])
t1 = time.time()
print("Training ended at", datetime.datetime.now())
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))


# # Naive Bayes

# In[ ]:


from sklearn.linear_model import LogisticRegression
import datetime, time, json
print("Starting training at", datetime.datetime.now())
t0 = time.time()


NB_reg = linear_model.BayesianRidge(n_iter=600)

NB_reg.fit(train_features, y_label)
Y_pred_nb_reg = NB_reg.predict(dev_features)

for i in range(2419):
    print(Y_pred_nb_reg[i])
t1 = time.time()
print("Training ended at", datetime.datetime.now())
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))


# # KNN

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
print("Starting training at", datetime.datetime.now())
t0 = time.time()

knn = KNeighborsRegressor(n_neighbors=2)
knn.fit(train_features, y_label) 
Y_pred_knn_reg = knn.predict(dev_features)

for i in range(2419):
    print(Y_pred_knn_reg[i])
t1 = time.time()
print("Training ended at", datetime.datetime.now())
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))


# # Decision Tree

# In[ ]:


# import the regressor 
from sklearn.tree import DecisionTreeRegressor
print("Starting training at", datetime.datetime.now())
t0 = time.time()

DT = DecisionTreeRegressor(random_state = 0, max_depth=22)

  
# fit the regressor with X and Y data 
DT.fit(train_features, y_label) 
Y_pred_DT_reg = DT.predict(dev_features)  
for i in range(2419):
    print(Y_pred_DT_reg[i])
t1 = time.time()
print("Training ended at", datetime.datetime.now())
print("Minutes elapsed: %f" % ((t1 - t0) / 60.)) 


# ### Gradient Boosting

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
print("Starting training at", datetime.datetime.now())
t0 = time.time()

GB = GradientBoostingRegressor()
#GB.fit(train_features, y_label) 
#Y_pred_GB_reg = GB.predict(dev_features)
param_dist = {"learning_rate": np.linspace(0.05, 0.15,5),
               "max_depth": range(3, 5),
               "min_samples_leaf": range(3, 5)}

rand = RandomizedSearchCV(GB, param_dist, cv=7,n_iter=10, random_state=5)
rand.fit(train_features, y_label)
print("Best Parameters : \n", rand.best_params_)

'''
for i in range(2419):
  print(Y_pred_GB_reg[i])
t1 = time.time()
print("Training ended at", datetime.datetime.now())
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))
'''


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
print("Starting training at", datetime.datetime.now())
t0 = time.time()

GB = GradientBoostingRegressor(min_samples_leaf= 3, max_depth= 4, learning_rate= 0.05)
GB.fit(train_features, y_label) 
Y_pred_GB_reg = GB.predict(dev_features)

for i in range(2419):
    print(Y_pred_GB_reg[i])
t1 = time.time()
print("Training ended at", datetime.datetime.now())
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))


# In[ ]:



GBReg = '\n'.join([str(elem) for elem in Y_pred_GB_reg]) 
len(GBReg)


# In[ ]:


from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
uploaded = drive.CreateFile({'title': 'GBRegresults.csv'})
uploaded.SetContentString(GBReg)
uploaded.Upload()
print('Uploaded file with ID {}'.format(uploaded.get('id')))
downloaded = drive.CreateFile({'id': uploaded.get('id')})
print('Downloaded content "{}"'.format(downloaded.GetContentString()))

