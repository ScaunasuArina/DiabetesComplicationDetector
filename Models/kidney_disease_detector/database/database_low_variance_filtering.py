import pandas as pd
from sklearn.preprocessing import normalize

# ==============================================
#             Low Variance Filter
# ==============================================

data = pd.read_csv('kidney_disease_formatted.csv')
print(f"SHAPE:\n{data.shape}\n\n")
print(f"HEAD:\n{data.head()}\n\n")
print(f"\nINFO:\n{data.info()}\n\n")
print(f"DESCRIPTION:\n{data.describe(include='all')}\n\n")

# sepparating class attribute from input attributes for classification algorithms
X = data.drop(['classification'], axis = 1)
y = data['classification']

print(f"X.columns: {X.columns}\n\n")

# before we calculate the variance of each variable, we need to make normalization
normalize = normalize(X)
X_scaled = pd.DataFrame(normalize)
print(f"X_scaled.var:{X_scaled.var()}\n")
print(f"X_scaled SHAPE:\n{X_scaled.shape}\n\n")

# storing the variance and name of variables
variance = X_scaled.var()
columns = X.columns

# saving the names of variables having variance more than a threshold value
variable = [ ]

for i in range(0,len(variance)):
    # if variance[i]>=0.000000003: # we keep 17/25 attributes with this threshold
    # if variance[i]>=0.000000004: # we keep 14/25 attributes with this threshold
    # if variance[i]>=0.00000001: # we keep 13/25 attributes with this threshold
    if variance[i]>=0.0000001: # we keep 10/25 attributes with this threshold
        variable.append(columns[i])

print(f"Variables list legth: {len(variable)}\n")
print(f"Variables: {variable}\n\n")

# creating a new dataframe using the above variables
X = X[variable]
print(f"X SHAPE: {X.shape}\n")
print(f"y SHAPE: {y.shape}")

# need to add back class attribute to dataframe
X['Classification'] = y
print(f"After concatenating both datas: -> X SHAPE: {X.shape}\n")

# rename all columns
X = X.rename(columns = {"age": "age",
                        "bp": "blood_pressure",
                        "bgr": "blood_glucose_random",
                        "bu": "blood_urea",
                        "sc": "serum_creatine",
                        "sod": "sodium",
                        "pot": "potassium",
                        "hemo": "hemoglobin",
                        "pcv": "packed_cell_volume",
                        "wc": "white_blood_cell_count",
                        "Classification": "classification"})

# Save the formatted file
X.to_csv('kidney_disease_low_variance_filter.csv', index=False)
