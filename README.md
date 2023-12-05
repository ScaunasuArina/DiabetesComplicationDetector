# DiabetesComplicationDetector

This project contains an Android application that uses pre-trained Machine Learning models to predict the probability of diabetes-related complications, such as renal or cardiovascular diseases. These Machine Learning models are called for prediction using the REST API server. This repository is part of the project along with the "AndroidDiabetesComplicationApp" repository.

Implementation of the REST API server and pre-trained models can be found this repo.

**Project diagram:**

![conectare android-python-EN](https://github.com/ScaunasuArina/DiabetesComplicationDetector/assets/44116228/f31cc014-bae7-4143-8404-050718c9e896)

Please note that this repo contains multiple Machine Learning models for every database. Performances of every model were analysed and the best model was chosen for each section ("disease").

**Links for databases:**

Diabetes Disease: https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system    
                  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;OR   
                  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;https://www.cdc.gov/brfss/annual_data/annual_data.htm  
                  
Heart Disease:    https://archive.ics.uci.edu/ml/datasets/Heart+Disease  
                  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;OR   
                  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;https://www.kaggle.com/datasets/rishidamarla/heart-disease-prediction  
                
Kidney Disease:   https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease  
                  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;OR  
                  &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;https://www.kaggle.com/datasets/mansoordaku/ckdisease  
