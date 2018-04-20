
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
import seaborn as sb

dataframe1 = pd.read_csv('C:/Users/yashas/LS_Dataset/diabetic_data.csv')


# In[14]:


print dataframe1.columns.values
#dataframe1.head(5)
#dataframe1.count


# In[15]:


#deleting nominal attributes

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

print dataframe1.columns.values


# In[16]:


#recoding y to yes or no from 3 cats

dataframe1['readmitted'] = pd.Series([0 if val == 'NO' 
                                     else 1
                                     for val in dataframe1['readmitted']])
print(dataframe1['readmitted'])


# In[17]:


#replace question marks with Nan

dataframe1 = dataframe1[dataframe1['race']!= '?']
dataframe1 = dataframe1[dataframe1['diag_1']!= '?']

#attributes = ['race','diag_1']

#for a in attributes:
#    x.loc[x[a] == '?', a] = np.NaN

#print x.diag_1[:1008]


# In[18]:


dataframe1[17:23]


# In[19]:


dataframe1['diag_1'] = pd.Series([1 if val.startswith('250')
                                 else 0
                                 for val in dataframe1['diag_1']], index = dataframe1.index)


# In[20]:


dataframe1['diag_1']


# In[21]:


#
dataframe1['number_outpatient'] = dataframe1['number_outpatient'].apply(lambda x:np.sqrt(x+0.5))
dataframe1['number_inpatient'] = dataframe1['number_inpatient'].apply(lambda x:np.sqrt(x+0.5))
dataframe1['number_emergency'] = dataframe1['number_emergency'].apply(lambda x:np.sqrt(x+0.5))


# In[22]:


#dataframe1['number_inpatient']


# In[23]:


scale_attr = ['time_in_hospital',
             'num_lab_procedures','num_procedures','num_medications',
             'number_outpatient','number_emergency','number_inpatient',
             'number_diagnoses']

from sklearn import preprocessing
my_scaler = preprocessing.StandardScaler().fit(dataframe1[scale_attr])
scale_data = my_scaler.transform(dataframe1[scale_attr])

scale_dataframe = pd.DataFrame(data = scale_data, columns = scale_attr, index = dataframe1.index )
dataframe1.drop(scale_attr, axis = 1, inplace = True)
dataframe1 = pd.concat([dataframe1, scale_dataframe], axis = 1)


# In[24]:


dataframe1['number_outpatient']


# In[25]:


#dataframe1.count
y = dataframe1['readmitted']
x = dataframe1.loc[:, dataframe1.columns != 'readmitted']


# In[26]:


#y
x.head(5)


# In[27]:


y


# In[28]:


#converting yes no columns to boolean

x.loc[x.change == 'Ch', 'change'] = 1
x.loc[x.change == 'No', 'change'] = 0


x.loc[x.diabetesMed == 'No', 'diabetesMed'] = 0
x.loc[x.diabetesMed == 'Yes', 'diabetesMed'] = 1

#x['change']


# In[29]:


x['change']


# In[30]:


cat_attributes = ['race','gender','age','admission_type_id','discharge_disposition_id','admission_source_id',
                  'max_glu_serum','A1Cresult',
                  'metformin','glimepiride','glipizide','glyburide','pioglitazone',
                  'rosiglitazone','insulin']

for i in cat_attributes: 
    x = pd.get_dummies(x, prefix = [i], columns = [i])
    


# In[31]:


x


# In[34]:


dataframe = x


# In[35]:


dataframe


# In[41]:


dataframe.to_csv('PPData1.csv')


# In[50]:


dataframe.iloc[:,0:3]

