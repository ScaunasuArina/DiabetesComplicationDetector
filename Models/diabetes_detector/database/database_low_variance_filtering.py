import pandas as pd
from sklearn.preprocessing import normalize

# ==============================================
#             Low Variance Filter
# ==============================================

data = pd.read_csv('diabetes_disease_formatted.csv')
print(f"SHAPE:\n{data.shape}\n\n")
print(f"HEAD:\n{data.head()}\n\n")
print(f"\nINFO:\n{data.info()}\n\n")
print(f"DESCRIPTION:\n{data.describe(include='all')}\n\n")

# separating class attribute from input attributes
X = data.drop(['Diabetes'], axis = 1)
y = data['Diabetes']

print(f"X columns: {X.columns}")

# before we calculate the variance of each variable, we need to make normalization
normalize = normalize(X)
X_scaled = pd.DataFrame(normalize)
print(f"X_scaled var: {X_scaled.var()}\n")
print(f"X_scaled SHAPE: {X_scaled.shape}\n")

# store the variance and name of variables
variance = X_scaled.var()
columns = X.columns

# save the names of variables having variance more than a threshold value
variable = []

for i in range(0,len(variance)):
    if variance[i]>=0.0002: # we keep 13/21 attributes with this threshold
    # if variance[i]>=0.0003: # we keep 7/21 attributes with this threshold
        variable.append(columns[i])

print(f"Variable len: {len(variable)}\n")
print(f"Variable: {variable}\n\n")

# creating a new dataframe using the above variables
X = X[variable]
print(f"X SHAPE: {X.shape}\n")
print(f"y SHAPE: {y.shape}")

# need to add back class attribute to dataframe
X['Classification'] = y
print(f"After concatenating both datas: -> X SHAPE: {X.shape}\n")

# rename all columns
X = X.rename(columns = {"HighBP": "blood_pressure",
                        "HighChol": "cholestrol",
                        "BMI": "bmi",
                        "Smoker": "smoker",
                        "PhysActivity": "physical_activity",
                        "Fruits": "fruits",
                        "GenHlth": "general_health",
                        "MentHlth": "mental_health",
                        "PhysHlth": "physical_health",
                        "Sex": "sex",
                        "Age": "age",
                        "Education": "education",
                        "Income": "income",
                        "Classification": "classification"})

# Save the formatted data to a new CSV file
X.to_csv('diabetes_disease_low_variance_filter.csv', index=False)
