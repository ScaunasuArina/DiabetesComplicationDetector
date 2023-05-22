import Models.diabetes_detector.database
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
import time

# ================================================
# See how to remove this part:

import pandas as pd
from sklearn.preprocessing import normalize

data = pd.read_csv('../database/diabetes_disease_formatted.csv')
print(f"SHAPE:\n{data.shape}\n\n")
print(f"HEAD:\n{data.head()}\n\n")
print(f"\nINFO:\n{data.info()}\n\n")
print(f"DESCRIPTION:\n{data.describe(include='all')}\n\n")

# Might need to drop some columns as we have 22 attributes
# separating class attribute from input attributes

# X = data.drop(['classification', 'sg', 'appet', 'rc', 'pcv', 'hemo', 'sod'], axis = 1)
X = data.drop(['Diabetes'], axis = 1)
y = data['Diabetes']

print(f"X columns: {X.columns}")

# Use Low Variance Filter to drop columns

#before we calculate the variance of each variable, we need to make normalization
normalize = normalize(X)
X_scaled = pd.DataFrame(normalize)
print(f"X_scaled var: {X_scaled.var()}\n")
print(f"X_scaled SHAPE: {X_scaled.shape}\n")

# storing the variance and name of variables
variance = X_scaled.var()
columns = X.columns

# saving the names of variables having variance more than a threshold value
variable = [ ]

for i in range(0,len(variance)):
    if variance[i]>=0.0002: # we keep 13/21 attributes with this threshold
    # if variance[i]>=0.0003: # we keep 7/21 attributes with this threshold
        variable.append(columns[i])

print(f"Variable len: {len(variable)}\n")
print(f"Variable: {variable}\n\n")

# creating a new dataframe using the above variables
X = X[variable]
print(f"X SHAPE: {X.shape}\n\n")


# Sepparate the data in train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 42)


# ==============================================

# svm_model = svm.SVC(kernel='rbf')

svm_model = svm.SVC(kernel='linear')

#coef_ = n_classes * (n_classes - 1) / 2
#coef_ -> This is only available in the case of a linear kernel.

# svm_model = svm.SVC(kernel='poly')

# gamma : {'scale', 'auto'} or float, default='scale'
# Kernel coefficient for 'rbf', 'poly' and 'sigmoid'

# svm_model = svm.SVC(kernel='poly', gamma='auto')

# svm_model = svm.SVC(kernel='sigmoid')

# (self, *, C=1.0, kernel='rbf', degree=3, gamma='scale',
# coef0=0.0, shrinking=True, probability=False,
# tol=1e-3, cache_size=200, class_weight=None,
# verbose=False, max_iter=-1, decision_function_shape='ovr',
# break_ties=False,
# random_state=None):
print("\nFitting the model...")

start_time = time.time()
svm_model.fit(X_train, y_train)
stop_time = time.time()

print(f"Start time: {start_time}\n")
print(f"Stop time: {stop_time}\n")
print(f"Training duration: {stop_time - start_time} seconds.")

model_predict = svm_model.predict(X_test)

print(f"CONFUSION MATRIX: {confusion_matrix(y_test, model_predict)}\n")
print(f"Accuracy is {round(accuracy_score(y_test, model_predict)*100, 2)}%\n")

