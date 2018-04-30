from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB

# Required Python Machine learning Packages
import pandas as pd
import numpy as np
# For preprocessing the data
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
# To split the dataset into train and test datasets
from sklearn.cross_validation import train_test_split
# To model the Gaussian Navie Bayes classifier
from sklearn.naive_bayes import GaussianNB
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn.neural_network import MLPClassifier

pp_df = pd.read_csv(
    "PPData.csv"
)

y = pp_df['readmitted']
X = pp_df.drop(['readmitted'], axis=1)

print(X.shape)
print(y.shape)

from sklearn.cross_validation import train_test_split
X_cv, X_test, y_cv, y_test = train_test_split(X, y, test_size=0.20)



# ============================================================================

##
## KNN
##

neigh = KNeighborsClassifier(n_neighbors=10 , algorithm='auto')
neigh.fit(X_cv, y_cv)
y_pred = neigh.predict(X_test)
print("Accuracy for KNN: ", accuracy_score(y_test, y_pred, normalize=True))
print("             ---")

# ============================================================================

##
##    Gausian Naive Bayes
##

clf = GaussianNB()
clf.fit(X_cv, y_cv)
y_pred = clf.predict(X_test)

print("Accuracy for Gausian Naive Bayes: " , accuracy_score(y_test, y_pred, normalize=True))
print("             --------------------")

# ============================================================================

#
#    SGD
#

clf = SGDClassifier(loss="log", penalty="elasticnet")
clf.fit(X_cv, y_cv)
y_pred = clf.predict(X_test)

print("Accuracy for SGD: ", accuracy_score(y_test, y_pred, normalize=True))
print("             ---")

# ============================================================================

#
#    Bernoulli NB
#

clf = BernoulliNB()
clf.fit(X_cv, y_cv)
y_pred = clf.predict(X_test)

print("Accuracy for Bernoulli NB: ",
      accuracy_score(y_test, y_pred, normalize=True))  # 60% accuracy
print("             ------------")

# ============================================================================

#
#    DecisionTree Classifier
#

clf = tree.DecisionTreeClassifier()
clf.fit(X_cv, y_cv)
y_pred = clf.predict(X_test)

print("Accuracy for Decision Tree Classifier: ",
      accuracy_score(y_test, y_pred, normalize=True))  # 60% accuracy
print("             ------------------------")

# ============================================================================

#
#    DecisionTree Classifier
#

clf = MLPClassifier(
    hidden_layer_sizes=(15,), random_state=1, max_iter=3, warm_start=True
)
clf.fit(X_cv, y_cv)
y_pred = clf.predict(X_test)

print("Accuracy for MLPClassifier: ",
      accuracy_score(y_test, y_pred, normalize=True))  # 60% accuracy
print("             -------------")
