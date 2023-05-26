import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('diabetes_disease.csv')
print(f"SHAPE:\n{data.shape}\n\n")
print(f"HEAD:\n{data.head()}\n\n")
print(f"\nINFO:\n{data.info()}\n\n")
print(f"DESCRIPTION:\n{data.describe(include='all')}\n\n")

# select specific columns
data_selected = data[[  'DIABETE3',
                        '_RFHYPE5',
                        'TOLDHI2', '_CHOLCHK',
                        '_BMI5',
                        'SMOKE100',
                        'CVDSTRK3', '_MICHD',
                        '_TOTINDA',
                        '_FRTLT1', '_VEGLT1',
                        '_RFDRHV5',
                        'HLTHPLN1', 'MEDCOST',
                        'GENHLTH', 'MENTHLTH', 'PHYSHLTH', 'DIFFWALK',
                        'SEX', '_AGEG5YR', 'EDUCA', 'INCOME2' ]]

print(f"SHAPE:{data_selected.shape}\n\n")
print(f"HEAD:{data_selected.head()}\n\n")

# drop missing values
data_selected = data_selected.dropna()
print(f"SHAPE SELECTED DATA:{data_selected.shape}\n\n")

# Modify and clean the values to be more suitable to ML algorithms

# DIABETE3
# make this attribute this ordinal
# 0 is for no diabetes or only during pregnancy
# 1 is for pre-diabetes or borderline diabetes
# 2 is for yes diabetes
# Remove all 7 (doesn't know)
# Remove all 9 (refused)
data_selected['DIABETE3'] = data_selected['DIABETE3'].replace({2:0, 3:0, 1:2, 4:1})
data_selected = data_selected[data_selected.DIABETE3 != 7]
data_selected = data_selected[data_selected.DIABETE3 != 9]
print(f'DIABETE3 unique: {data_selected.DIABETE3.unique()}\n\n')

print(f"Presence diabetes values: {sum(data_selected['DIABETE3'] == 2.0)}\n")
print(f"Absence diabetes values: {sum(data_selected['DIABETE3'] == 0.0)}\n\n")

# 1 _RFHYPE5
#Change 1 to 0, so it represents No high blood pressure and
#Change 2 to 1, so it represents high blood pressure
data_selected['_RFHYPE5'] = data_selected['_RFHYPE5'].replace({1:0, 2:1})
data_selected = data_selected[data_selected._RFHYPE5 != 9]
print(f'_RFHYPE5 unique: {data_selected._RFHYPE5.unique()}\n\n')

# 2 TOLDHI2
# Change 2 to 0 because it is 'No'
# Remove all 7 (doesn't know)
# Remove all 9 (refused)
data_selected['TOLDHI2'] = data_selected['TOLDHI2'].replace({2:0})
data_selected = data_selected[data_selected.TOLDHI2 != 7]
data_selected = data_selected[data_selected.TOLDHI2 != 9]
print(f'TOLDHI2 unique: {data_selected.TOLDHI2.unique()}\n\n')

# 3 _CHOLCHK
# Change 3 to 0 and 2 to 0 for Not checked cholesterol in past 5 years
# Remove 9
data_selected['_CHOLCHK'] = data_selected['_CHOLCHK'].replace({3:0,2:0})
data_selected = data_selected[data_selected._CHOLCHK != 9]
print(f'_CHOLCHK unique: {data_selected._CHOLCHK.unique()}\n\n')

# 4 _BMI5 (no changes, just note that these are BMI * 100. So for example a BMI of 4018 is really 40.18)
data_selected['_BMI5'] = data_selected['_BMI5'].div(100).round(0)
print(f'_BMI5 unique: {data_selected._BMI5.unique()}\n\n')

# 5 SMOKE100
# Change 2 to 0 because it is 'No'
# Remove all 7 (dont knows)
# Remove all 9 (refused)
data_selected['SMOKE100'] = data_selected['SMOKE100'].replace({2:0})
data_selected = data_selected[data_selected.SMOKE100 != 7]
data_selected = data_selected[data_selected.SMOKE100 != 9]
data_selected.SMOKE100.unique()
print(f'SMOKE100 unique: {data_selected.SMOKE100.unique()}\n\n')

# 6 CVDSTRK3
# Change 2 to 0 because it is No
# Remove all 7 (dont knows)
# Remove all 9 (refused)
data_selected['CVDSTRK3'] = data_selected['CVDSTRK3'].replace({2:0})
data_selected = data_selected[data_selected.CVDSTRK3 != 7]
data_selected = data_selected[data_selected.CVDSTRK3 != 9]
print(f'CVDSTRK3 unique: {data_selected.CVDSTRK3.unique()}\n\n')

# 7 _MICHD
#Change 2 to 0 because this means did not have MI or CHD
data_selected['_MICHD'] = data_selected['_MICHD'].replace({2: 0})
data_selected._MICHD.unique()
print(f'_MICHD unique: {data_selected._MICHD.unique()}\n\n')

# 8 _TOTINDA
# 1 for physical activity
# change 2 to 0 for no physical activity
# Remove all 9 (don't know/refused)
data_selected['_TOTINDA'] = data_selected['_TOTINDA'].replace({2:0})
data_selected = data_selected[data_selected._TOTINDA != 9]
print(f'_TOTINDA unique: {data_selected._TOTINDA.unique()}\n\n')

# 9 _FRTLT1
# Change 2 to 0. this means no fruit consumed per day. 1 will mean consumed 1 or more pieces of fruit per day
# remove all dont knows and missing 9
data_selected['_FRTLT1'] = data_selected['_FRTLT1'].replace({2:0})
data_selected = data_selected[data_selected._FRTLT1 != 9]
print(f'_FRTLT1 unique: {data_selected._FRTLT1.unique()}\n\n')

# 10 _VEGLT1
# Change 2 to 0. this means no vegetables consumed per day. 1 will mean consumed 1 or more pieces of vegetable per day
# remove all dont knows and missing 9
data_selected['_VEGLT1'] = data_selected['_VEGLT1'].replace({2:0})
data_selected = data_selected[data_selected._VEGLT1 != 9]
print(f'_VEGLT1 unique: {data_selected._VEGLT1.unique()}\n\n')

# 11 _RFDRHV5
# Change 1 to 0 (1 was no for heavy drinking). change all 2 to 1 (2 was yes for heavy drinking)
# remove all dont knows and missing 9
data_selected['_RFDRHV5'] = data_selected['_RFDRHV5'].replace({1:0, 2:1})
data_selected = data_selected[data_selected._RFDRHV5 != 9]
print(f'_RFDRHV5 unique: {data_selected._RFDRHV5.unique()}\n\n')

# 12 HLTHPLN1
# 1 is yes, change 2 to 0 because it is No health care access
# remove 7 and 9 for don't know or refused
data_selected['HLTHPLN1'] = data_selected['HLTHPLN1'].replace({2:0})
data_selected = data_selected[data_selected.HLTHPLN1 != 7]
data_selected = data_selected[data_selected.HLTHPLN1 != 9]
print(f'HLTHPLN1 unique: {data_selected.HLTHPLN1.unique()}\n\n')

# 13 MEDCOST
# Change 2 to 0 for no, 1 is already yes
# remove 7 for don/t know and 9 for refused
data_selected['MEDCOST'] = data_selected['MEDCOST'].replace({2:0})
data_selected = data_selected[data_selected.MEDCOST != 7]
data_selected = data_selected[data_selected.MEDCOST != 9]
data_selected.MEDCOST.unique()
print(f'MEDCOST unique: {data_selected.MEDCOST.unique()}\n\n')

# 14 GENHLTH
# This is an ordinal variable that I want to keep (1 is Excellent -> 5 is Poor)
# Remove 7 and 9 for don't know and refused
data_selected = data_selected[data_selected.GENHLTH != 7]
data_selected = data_selected[data_selected.GENHLTH != 9]
print(f'GENHLTH unique: {data_selected.GENHLTH.unique()}\n\n')

# 15 MENTHLTH
# already in days so keep that, scale will be 0-30
# change 88 to 0 because it means none (no bad mental health days)
# remove 77 and 99 for don't know not sure and refused
data_selected['MENTHLTH'] = data_selected['MENTHLTH'].replace({88:0})
data_selected = data_selected[data_selected.MENTHLTH != 77]
data_selected = data_selected[data_selected.MENTHLTH != 99]
print(f'MENTHLTH unique: {data_selected.MENTHLTH.unique()}\n\n')

# 16 PHYSHLTH
# already in days so keep that, scale will be 0-30
# change 88 to 0 because it means none (no bad mental health days)
# remove 77 and 99 for don't know not sure and refused
data_selected['PHYSHLTH'] = data_selected['PHYSHLTH'].replace({88:0})
data_selected = data_selected[data_selected.PHYSHLTH != 77]
data_selected = data_selected[data_selected.PHYSHLTH != 99]
print(f'PHYSHLTH unique: {data_selected.PHYSHLTH.unique()}\n\n')

# 17 DIFFWALK
# change 2 to 0 for no. 1 is already yes
# remove 7 and 9 for don't know not sure and refused
data_selected['DIFFWALK'] = data_selected['DIFFWALK'].replace({2:0})
data_selected = data_selected[data_selected.DIFFWALK != 7]
data_selected = data_selected[data_selected.DIFFWALK != 9]
print(f'DIFFWALK unique: {data_selected.DIFFWALK.unique()}\n\n')

# 18 SEX
# in other words - is respondent male (somewhat arbitrarily chose this change because men are at higher risk for heart disease)
# change 2 to 0 (female as 0). Male is 1
data_selected['SEX'] = data_selected['SEX'].replace({2:0})
print(f'SEX unique: {data_selected.SEX.unique()}\n\n')

# 19 _AGEG5YR
# already ordinal. 1 is 18-24 all the way up to 13 wis 80 and older. 5 year increments.
# remove 14 because it is don't know or missing
data_selected = data_selected[data_selected._AGEG5YR != 14]
data_selected._AGEG5YR.unique()
print(f'_AGEG5YR: {data_selected._AGEG5YR.unique()}\n\n')

# 20 EDUCA
# This is already an ordinal variable with 1 being never attended school or kindergarten only up to 6 being college 4 years or more
# Scale here is 1-6
# Remove 9 for refused
data_selected = data_selected[data_selected.EDUCA != 9]
print(f'EDUCA: {data_selected.EDUCA.unique()}\n\n')

# 21 INCOME2
# Variable is already ordinal with 1 being less than $10,000 all the way up to 8 being $75,000 or more
# Remove 77 and 99 for don't know and refused
data_selected = data_selected[data_selected.INCOME2 != 77]
data_selected = data_selected[data_selected.INCOME2 != 99]
print(f'INCOME2: {data_selected.INCOME2.unique()}\n\n')

# check the shape of the dataset now: We have 253,680 cleaned rows and 22 columns (1 of which is our dependent variable)
print(f"SHAPE SELECTED DATA:{data_selected.shape}\n\n")
print(f"HEAD SELECTED DATA: {data_selected.head(10)}\n\n")

# check Class Sizes of the heart disease column
print(f"CLASS SIZE:\n{data_selected.groupby(['DIABETE3']).size()}\n\n")

# rename the columns to make them more readable
data = data_selected.rename(columns = {'DIABETE3':'Diabetes_012',
                                            '_RFHYPE5':'HighBP',
                                            'TOLDHI2':'HighChol', '_CHOLCHK':'CholCheck',
                                            '_BMI5':'BMI',
                                            'SMOKE100':'Smoker',
                                            'CVDSTRK3':'Stroke', '_MICHD':'HeartDiseaseorAttack',
                                            '_TOTINDA':'PhysActivity',
                                            '_FRTLT1':'Fruits', '_VEGLT1':"Veggies",
                                            '_RFDRHV5':'HvyAlcoholConsump',
                                            'HLTHPLN1':'AnyHealthcare', 'MEDCOST':'NoDocbcCost',
                                            'GENHLTH':'GenHlth', 'MENTHLTH':'MentHlth', 'PHYSHLTH':'PhysHlth', 'DIFFWALK':'DiffWalk',
                                            'SEX':'Sex', '_AGEG5YR':'Age', 'EDUCA':'Education', 'INCOME2':'Income' })

# Check how many respondents have no diabetes, prediabetes or diabetes. Note the class imbalance!
# 0 - non-diabetes
# 1 - pre-diabetes
# 2 - diabetes
print(f"CLASS SIZE:\n{data.groupby(['Diabetes_012']).size()}\n\n")

# check heatmap
import seaborn as sns
plt.figure(figsize = (20,20))
sns.heatmap(data.corr(), annot = True, fmt=".2f",linewidths=0.5)

print(f"DATA CORELATION: {data.corr()}\n\n")

# Need to drop some entries as we have 213k non-diabetes and 35k diabetes, therefore an imbalance.
# select all diabetes entries
diabetes_df = data[data.Diabetes_012 == 2.0]
# select only the first 500 diabetes entires
diabetes_df = diabetes_df.head(35000)
diabetes_df.groupby(["Diabetes_012"]).size()

# select all non-diabetes entries
non_diabetes_df = data[data.Diabetes_012 == 0.0]
# select only the first 500 entries
non_diabetes_df = non_diabetes_df.head(35000)
non_diabetes_df.groupby(["Diabetes_012"]).size()

# create the final dataframe used for classification
concat_df = [diabetes_df, non_diabetes_df]
final_data = pd.concat(concat_df)

# shuffle entries from the final dataframe
final_data = final_data.sample(frac=1).reset_index(drop=True)
print(f"FINAL DATA SHAPE: {final_data.shape}\n\n")
print(f"FINAL DATA HEAD:\n{final_data.head()}\n\n")

# rename diabetes attribute to be more reliable
final_data = final_data.rename(columns = {'Diabetes_012':'Diabetes'})

print(f"\nINFO:\n{final_data.info()}\n\n")

# Save the formatted data to a new CSV file
data.to_csv('diabetes_disease_formatted.csv', index=False)
