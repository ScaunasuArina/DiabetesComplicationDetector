import numpy as np
import pandas as pd

data = pd.read_csv('kidney_disease.csv')
print(f"SHAPE:\n{data.shape}\n\n")
print(f"HEAD:\n{data.head()}\n\n")
print(f"\nINFO:\n{data.info()}\n\n")
print(f"DESCRIPTION:\n{data.describe(include='all')}\n\n")

print(f"Presence Chronic Kidney disease values: {sum(data['classification'] == 1)}")
print(f"Absence Chronic Kidney disease values: {sum(data['classification'] == 0)}")

NumericalColumns = ['age','bp','al','su','bgr','bu','sc','sod','pot','hemo']
CategoricalColumns = ['rbc','sg','pc','pcc','ba','pcv','wc','rc','htn','dm','cad','appet','pe','ane']

# verify missing values
print(f"Missing values:\n{data.isnull().sum()}\n\n")

# check all attributes
print(f'Classification unique: {data.classification.unique()}\n')
print(f'Age unique: {data.age.unique()}\n')
print(f'bp unique: {data.bp.unique()}\n')
print(f'sg unique: {data.sg.unique()}\n')
print(f'al unique: {data.al.unique()}\n')
print(f'su unique: {data.su.unique()}\n')
print(f'rbc unique: {data.rbc.unique()}\n')
print(f'pc unique: {data.pc.unique()}\n')
print(f'pcc unique: {data.pcc.unique()}\n')
print(f'ba unique: {data.ba.unique()}\n')

# adjust values for all attributes
data['classification'].replace("ckd\t","ckd",inplace=True)
data['dm'].replace(["\tno","\tyes"," yes"],["no","yes","yes"],inplace=True)
data['cad'].replace(["\tno"],["no"],inplace=True)
data['rc'].replace("\t?",np.nan, inplace=True)
data.wc.replace("\t?",np.nan, inplace=True)
data['wc'] = data['wc'].replace(["\t6200","\t8400"],[6200,8400])
data['pcv'].replace(["\t?","\t43"],np.nan, inplace=True)

# fill missing values
for columnName in CategoricalColumns:
    data[columnName].fillna(data[columnName].mode()[0], inplace=True)
for columnName in NumericalColumns:
    data[columnName].fillna(data[columnName].mean(), inplace=True)

# verify missing values
print(f"Missing values:\n{data.isnull().sum()}\n\n")

encodeColumn = ['rbc','pc' ,'pcc' ,'ba' ,'htn' ,'dm' ,'cad' ,'appet' ,'pe' ,'ane']
data = pd.get_dummies(data , columns=encodeColumn , prefix=encodeColumn , drop_first=True)

# replace classification attribute string values to binary ones
data['classification'].replace(["ckd","notckd"],[1,0], inplace=True)
print(f"Classification:\n{data.classification.value_counts()}\n\n")

print(f"\nINFO:\n{data.info()}\n\n")

# convert attributes from object type to int/float
data['pcv']=data['pcv'].astype(int)
data['wc']=data['wc'].astype(int)
data['rc']=data['rc'].astype(float)
print(f"\nINFO:\n{data.info()}\n\n")

# see which attributes that are set as type 'object'
object_dtypes = data.select_dtypes(include = 'object')
print(f"object_dtypes HEAD: {object_dtypes.head()}\n\n")

# convert string values to boolean values
string_2_boolean_values = {
                            "rbc": {
                                "abnormal":1,
                                "normal": 0,
                            },
                            "pc":{
                                "abnormal":1,
                                "normal": 0,
                            },
                            "pcc":{
                                "present":1,
                                "notpresent":0,
                            },
                            "ba":{
                                "notpresent":0,
                                "present": 1,
                            },
                            "htn":{
                                "yes":1,
                                "no": 0,
                            },
                            "dm":{
                                "yes":1,
                                "no":0,
                            },
                            "cad":{
                                "yes":1,
                                "no": 0,
                            },
                            "appet":{
                                "good":1,
                                "poor": 0,
                            },
                            "pe":{
                                "yes":1,
                                "no":0,
                            },
                            "ane":{
                                "yes":1,
                                "no":0,
                            }
                        }
data = data.replace(string_2_boolean_values)

print(f"HEAD:\n{data.head()}\n\n")
print(f"Presence Chronic Kidney disease values: {sum(data['classification'] == 1)}")
print(f"Absence Chronic Kidney disease values: {sum(data['classification'] == 0)}")

# should drop 'id' column because it gives no information about the disease
data = data.drop(['id'], axis = 1)

# Save the formatted data to a new CSV file
data.to_csv('kidney_disease_formatted.csv', index=False)

