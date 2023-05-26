import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('heart_disease.csv')
print(f"SHAPE:\n{data.shape}\n\n")
print(f"HEAD:\n{data.head()}\n\n")
print(f"INFO:\n{data.info()}\n\n")
print(f"DESCRIPTION:\n{data.describe(include='all')}\n\n")

#see how many unique values we have for 'classification' attribute
print(f"Unique values for classification attribute: {data.Heart_Disease.unique()}\n\n")

#change 'classification' values to binary  values
# 1 = ckd
# 0 = notckd
data['Heart_Disease'] = data['Heart_Disease'].replace(['Presence','Absence'], [1,0])

print(f"HEAD:\n{data.head(10)}\n\n")

# see how many null values we have for each attribute
print(f"NULL VALUES:\n{data.isnull().sum()}\n\n")

# check data type for each attribute
print(f"DATA TYPE:\n{data.info()}\n\n")

data.index = range(0,len(data),1)
print(f"INDEX:\n{data.index}\n\n")

# check heatmap
plt.figure(figsize = (20,20))
sns.heatmap(data.corr(), annot = True, fmt=".2f",linewidths=0.5)

# check data corelation
print(f"DATA CORELATION:\n{data.corr()}\n\n")

# rename all columns
data = data.rename(columns = {  "Age": "age",
                                "Sex": "sex",
                                "Chest_pain_type": "chest_pain",
                                "BP": "blood_pressure",
                                "Cholesterol": "cholestrol",
                                "FBS_over_120": "blood_sugar",
                                "EKG_results": "ecg",
                                "Max_HR": "heart_rate",
                                "Exercise_angina": "exercise",
                                "ST_depression": "old_peak",
                                "Slope_of_ST": "slope_of_st",
                                "Number_of_vessels_fluro": "no_vessels_fluro",
                                "Thallium": "thallium_scan",
                                "Heart_Disease": "classification"})

# Check classes distribution (seems to be almost equal)
print(f"Presence heart disease values: {sum(data['classification'] == 1)}")
print(f"Absence heart disease values: {sum(data['classification'] == 0)}")
print(f"New shape: {data.shape}")
print(f"HEAD: {data.head()}")

# Save the formatted data to a new CSV file
data.to_csv('heart_disease_formatted.csv', index=False)
