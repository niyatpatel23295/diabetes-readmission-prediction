# Ignore warnings
import warnings

# Handle table-like data and matrices
import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn.feature_selection import RFECV
# Modelling Helpers
from sklearn.preprocessing import Imputer, Normalizer, scale


warnings.filterwarnings('ignore')


from flask import Flask, render_template, request
app = Flask(__name__)


@app.route("/")
def test():
    return render_template('welcome.html')

@app.route("/predict", methods= ['GET', 'POST'])
def predict():
    print(request.form['email'])
    return  "Nne"

    # p_diabetic_data = pd.read_csv("dataset_diabetes/preview_diabetic_data.csv")
    # diabetic_data = pd.read_csv("dataset_diabetes/diabetic_data.csv")

    # # New Gender
    # new_gender = pd.Series(np.where(  diabetic_data.gender == "Male", 1, 0), name = 'gender')

    # # New Race
    # new_race = pd.get_dummies(diabetic_data.race, prefix="race")

    # new_race['race_Other'] = new_race['race_?'] + new_race['race_Other']

    # new_race = pd.DataFrame(new_race).drop(['race_?', 'race_Other'], axis =1)

    # # New Age
    # new_age = pd.DataFrame(p_diabetic_data['age']).replace({
    #     '[0-10)': 0,
    #     '[10-20)': 1,
    #     '[20-30)': 2,
    #     '[30-40)': 3,
    #     '[40-50)': 4,
    #     '[50-60)': 5,
    #     '[60-70)': 6,
    #     '[70-80)': 7,
    #     '[80-90)': 8,
    #     '[90-100)': 9,
    #     '[100-110)': 10,
    # })
