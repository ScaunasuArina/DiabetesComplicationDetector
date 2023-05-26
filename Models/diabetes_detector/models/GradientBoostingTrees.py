from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import time
import pandas as pd

data = pd.read_csv('../database/diabetes_disease_low_variance_filter.csv')
X = data.drop(['classification'], axis = 1)
y = data['classification']
# Separate the data in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 42)

# ==============================================
#         Gradient Boosting Trees Model
# ==============================================
# gradientBoosting_model = GradientBoostingClassifier()
# gradientBoosting_model = GradientBoostingClassifier(n_estimators=200)
# gradientBoosting_model = GradientBoostingClassifier(n_estimators=300)  #lower accuracy than 200 estimators

# gradientBoosting_model = GradientBoostingClassifier(n_estimators=200, criterion = 'mse') # exactly the same accuracy

gradientBoosting_model = GradientBoostingClassifier(n_estimators=200, max_depth=10)
print("\nFitting the model...")

start_time = time.time()
gradientBoosting_model.fit(X_train, y_train)
stop_time = time.time()

print(f"Start time: {start_time}")
print(f"Stop time: {stop_time}\n")
print(f"Training duration: {stop_time - start_time} seconds.")

model_predict = gradientBoosting_model.predict(X_test)

print(f"CONFUSION MATRIX: {confusion_matrix(y_test, model_predict)}\n")
print(f"Accuracy is {round(accuracy_score(y_test, model_predict)*100, 2)}%\n")
