import pandas as pd
from sklearn.preprocessing import normalize

data = pd.read_csv('diabetes_disease_formatted.csv')
print(f"SHAPE:\n{data.shape}\n\n")
print(f"HEAD:\n{data.head()}\n\n")
print(f"\nINFO:\n{data.info()}\n\n")
print(f"DESCRIPTION:\n{data.describe(include='all')}\n\n")

# TODO: not working to define this as a class ???
class Database:
    def __init__(self, X, y):
        self.X = X
        self.y = y

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
