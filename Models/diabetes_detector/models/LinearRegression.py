from sklearn.linear_model import LinearRegression
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
#                  Linear Regression Model
# ==============================================
linear_model = LinearRegression()
print("\nFitting the model...")

start_time = time.time()
linear_model.fit(X_train, y_train)
stop_time = time.time()

print(f"Start time: {start_time}")
print(f"Stop time: {stop_time}\n")
print(f"Training duration: {stop_time - start_time} seconds.\n\n")

r_sq = linear_model.score(X_test, y_test)
print(f"Coefficient of determination: {r_sq}")