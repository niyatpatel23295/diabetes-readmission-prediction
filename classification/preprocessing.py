# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import seaborn as sb

dataframe1 = pd.read_csv(
    '/Users/niyatpatel/Documents/Study/CMPE256/cmpe256 Project/dataset_diabetes/diabetic_data.csv'
)


print(dataframe1.age.unique())


# In[2]:

print(dataframe1.columns.values)
#dataframe1.head(5)
#dataframe1.count

# In[3]:

#deleting attributes

del dataframe1['encounter_id']
del dataframe1['patient_nbr']
del dataframe1['payer_code']

del dataframe1['weight']
del dataframe1['medical_specialty']

del dataframe1['diag_2']
del dataframe1['diag_3']

del dataframe1['repaglinide']
del dataframe1['nateglinide']
del dataframe1['chlorpropamide']
del dataframe1['acetohexamide']
del dataframe1['tolbutamide']
del dataframe1['acarbose']
del dataframe1['miglitol']
del dataframe1['troglitazone']
del dataframe1['tolazamide']
del dataframe1['examide']
del dataframe1['citoglipton']
del dataframe1['glyburide-metformin']
del dataframe1['glipizide-metformin']
del dataframe1['glimepiride-pioglitazone']
del dataframe1['metformin-rosiglitazone']
del dataframe1['metformin-pioglitazone']

print(dataframe1.columns.values)

# In[4]:

#recoding 0 to No readmission and 1 to <30 or >30 readmission from readmitted attribute

dataframe1['readmitted'] = pd.Series(
    [0 if val == 'NO' else 1 for val in dataframe1['readmitted']])
dataframe1['change'] = pd.Series(
    [0 if val == 'No' else 1 for val in dataframe1['change']])
dataframe1['diabetesMed'] = pd.Series(
    [0 if val == 'No' else 1 for val in dataframe1['diabetesMed']])

#print(dataframe1['readmitted'])
#'change','diabetesMed',

# In[5]:

#replace question marks with Nan

dataframe1 = dataframe1[dataframe1['race'] != '?']
dataframe1 = dataframe1[dataframe1['diag_1'] != '?']

#dataframe1[17:23]

# In[6]:

#Making categorical attribute to binary with 1 for diabetic category and 0 for others
dataframe1['diag_1'] = pd.Series(
    [1 if val.startswith('250') else 0 for val in dataframe1['diag_1']],
    index=dataframe1.index)

#dataframe1['diag_1']

# In[7]:

#
dataframe1['number_outpatient'] = dataframe1['number_outpatient'].apply(
    lambda x: np.sqrt(x + 0.5))
dataframe1['number_inpatient'] = dataframe1['number_inpatient'].apply(
    lambda x: np.sqrt(x + 0.5))
dataframe1['number_emergency'] = dataframe1['number_emergency'].apply(
    lambda x: np.sqrt(x + 0.5))

#dataframe1['number_inpatient']

# In[8]:

scale_attr = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses'
]

from sklearn import preprocessing
my_scaler = preprocessing.StandardScaler().fit(dataframe1[scale_attr])
scale_data = my_scaler.transform(dataframe1[scale_attr])

scale_dataframe = pd.DataFrame(
    data=scale_data, columns=scale_attr, index=dataframe1.index)
dataframe1.drop(scale_attr, axis=1, inplace=True)
dataframe1 = pd.concat([dataframe1, scale_dataframe], axis=1)

#dataframe1['number_outpatient']

# In[9]:

cat_attributes = [
    'race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
    'admission_source_id', 'max_glu_serum', 'A1Cresult', 'metformin',
    'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone',
    'insulin'
]

for i in cat_attributes:
    dataframe1 = pd.get_dummies(dataframe1, prefix=[i], columns=[i])

# In[ ]:

#dataframe1.count
#y = dataframe1['readmitted']
#x = dataframe1.loc[:, dataframe1.columns != 'readmitted']
#y
#x.head(5)
#dataframe1
#dataframe.iloc[:,0:3]

# In[10]:
print(dataframe1)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn.neural_network import MLPClassifier

dataframe1 = pd.read_csv("PPData.csv")

y = dataframe1['readmitted']
X = dataframe1.drop(['readmitted'], axis=1)

print(X.shape)
print(y.shape)

from sklearn.cross_validation import train_test_split
X_cv, X_test, y_cv, y_test = train_test_split(X, y, test_size=0.20)

# ============================================================================

##
## KNN
##

neigh = KNeighborsClassifier(n_neighbors=10, algorithm='auto')
neigh.fit(X_cv, y_cv)
y_pred = neigh.predict(X_test)
print("Accuracy for KNN: ", accuracy_score(y_test, y_pred, normalize=True))
print("             ---")