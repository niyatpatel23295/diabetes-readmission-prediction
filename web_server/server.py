# Ignore warnings
import warnings

# Handle table-like data and matrices
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn.feature_selection import RFECV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
# Modelling Helpers
from sklearn.preprocessing import Imputer, Normalizer, scale
from werkzeug.datastructures import ImmutableMultiDict
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')


from flask import Flask, render_template, request
app = Flask(__name__)


@app.route("/")
def test():
    return render_template('welcome.html')

@app.route("/predict", methods= ['GET', 'POST'])
def predict():
    with open(
            "/Users/niyatpatel/Documents/Study/CMPE256/cmpe256 Project/dataset_diabetes/diabetic_data.csv",
            "a") as myfile:
        myfile.write("\n" + request.form['data'] + ',No')

    dataframe1 = pd.read_csv(
        '/Users/niyatpatel/Documents/Study/CMPE256/cmpe256 Project/dataset_diabetes/diabetic_data.csv'
    )




    print(dataframe1)





    #in_df.head(5)
    #in_df.count

    # In[3]:

    # In[3]:

    print(dataframe1.columns.values)
    #dataframe1.head(5)
    #dataframe1.count

    # In[4]:

    #After many iterations these attributes are being removed
    #unique ids, won't contribute to classification

    del dataframe1['encounter_id']
    del dataframe1['patient_nbr']
    del dataframe1['payer_code']

    #Higher percentage of missing values
    del dataframe1['weight']
    del dataframe1['medical_specialty']

    #These won't affect the classification as the variance is too low
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

    # In[5]:

    #recoding 0 to No readmission and 1 to <30 or >30 readmission from readmitted attribute
    dataframe1['readmitted'] = pd.Series(
        [0 if val == 'NO' else 1 for val in dataframe1['readmitted']])

    # In[6]:

    #Direct binary cnversion
    dataframe1['change'] = pd.Series(
        [0 if val == 'No' else 1 for val in dataframe1['change']])
    dataframe1['diabetesMed'] = pd.Series(
        [0 if val == 'No' else 1 for val in dataframe1['diabetesMed']])

    #print(dataframe1['readmitted'])
    #'change','diabetesMed',

    # In[7]:

    #not consider rows in ? in dataframe or removing empty rows in race and diag_1

    dataframe1 = dataframe1[dataframe1['race'] != '?']
    dataframe1 = dataframe1[dataframe1['diag_1'] != '?']

    #dataframe1[17:23]

    # In[8]:

    dataframe1.shape

    # In[9]:

    #Making categorical attribute to binary with 1 for diabetic category and 0 for others
    dataframe1['diag_1'] = pd.Series(
        [1 if val.startswith('250') else 0 for val in dataframe1['diag_1']],
        index=dataframe1.index)

    #dataframe1['diag_1']

    # In[10]:

    #Square root transformation to reduce effects of extreme values
    dataframe1['number_outpatient'] = dataframe1['number_outpatient'].apply(
        lambda x: np.sqrt(x + 0.5))
    dataframe1['number_inpatient'] = dataframe1['number_inpatient'].apply(
        lambda x: np.sqrt(x + 0.5))
    dataframe1['number_emergency'] = dataframe1['number_emergency'].apply(
        lambda x: np.sqrt(x + 0.5))

    #dataframe1['number_inpatient']

    # In[11]:

    #standardisation to have 0 mean and unit variance
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

    # In[12]:

    #converting categorical attributes to numerical i.e., binary attributes
    cat_attributes = [
        'race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
        'admission_source_id', 'max_glu_serum', 'A1Cresult', 'metformin',
        'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone',
        'insulin'
    ]

    for i in cat_attributes:
        dataframe1 = pd.get_dummies(dataframe1, prefix=[i], columns=[i])

    test = dataframe1.tail(n=1)

    dataframe1.drop([len(dataframe1) - 1])

    y = dataframe1['readmitted']
    X = dataframe1.drop(['readmitted'], axis=1)

    X_test = test.drop(['readmitted'], axis=1)
    
    # from sklearn.cross_validation import train_test_split
    # X_cv, X_test, y_cv, y_test = train_test_split(X, y, test_size=  1 )

    #MLP classifier


    mlp = MLPClassifier()
    model = mlp.fit(X,y) 
    y_pred = model.predict(X_test)

    print(y_pred)
    if(y_pred[0] == 1):
        return "Yes, Patient might need to be Readmitted"
    else:
        return "No, Patient won't need to be readmitted!"
