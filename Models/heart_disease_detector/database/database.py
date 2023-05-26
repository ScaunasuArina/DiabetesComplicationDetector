import pandas as pd

data = pd.read_csv('heart_disease_formatted.csv')
print(f"SHAPE:\n{data.shape}\n\n")
print(f"\nHEAD:\n{data.head()}\n\n")

# separating class attribute from input attributes
y = data['Heart_Disease']
X = data.drop(['Heart_Disease'], axis = 1)

print(f"X columns: {X.columns}")