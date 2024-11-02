import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df.head()

# Shape of dataset
df.shape

### Check missing values

# Check for missing values
df.isna().sum()

### Handle Outliers

# Checking for outliers
df.boxplot()
plt.xticks(rotation=90)
plt.show()

# Handing outliers
outlier_colms = ['creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']
df1 = df.copy()

def handle_outliers(df, colm):
    '''Change the values of outlier to upper and lower whisker values '''
    q1 = df.describe()[colm].loc["25%"]
    q3 = df.describe()[colm].loc["75%"]
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    for i in range(len(df)):
        if df.loc[i,colm] > upper_bound:
            df.loc[i,colm]= upper_bound
        if df.loc[i,colm] < lower_bound:
            df.loc[i,colm]= lower_bound
    return df

for colm in outlier_colms:
    df1 = handle_outliers(df1, colm)

# Recheck for outliers
df1.boxplot()
plt.xticks(rotation=90)
plt.show()

### Split into training and testing set

# Split dataset into training and testing set, considering all features for prediction

X = df1.iloc[:, :-1].values
y = df1['DEATH_EVENT'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state= 123)

X_train[1]

### Model Training

xgb_clf = XGBClassifier(n_estimators=200, max_depth=4, max_leaves=5, random_state=42)
xgb_clf.fit(X_train, y_train)

### Model Performance

# Accuracy

train_acc = accuracy_score(y_train, xgb_clf.predict(X_train))
test_acc = accuracy_score(y_test, xgb_clf.predict(X_test))
print("Training accuracy: ", train_acc)
print("Testing accuracy: ", test_acc)

# F1-score

train_f1 = f1_score(y_train, xgb_clf.predict(X_train))
test_f1 = f1_score(y_test, xgb_clf.predict(X_test))
print("Training F1 score: ", train_f1)
print("Testing F1 score: ", test_f1)

### Save the trained model

# Prepare versioned save file name
save_file_name = "xgboost-model.pkl"

joblib.dump(xgb_clf, save_file_name)

## Gradio Implementation

# !pip -q install gradio

import gradio
import joblib
import numpy as np

# Load your trained model
model =  joblib.load('xgboost-model.pkl')
# YOUR CODE HERE

df1.head()

d1 = [1,24,4,5,6,4]


# Function for prediction

def predict_death_event(age, anemia,creatinine_phosphokinase,diabetes, ejection_fraction,
                        high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time):
  # YOUR CODE HERE...
  yes_no_mapping ={'Yes':1, 'No':0}
  sex_mapping = {'M':1,'F':0}

  input_data = [age, yes_no_mapping[anemia],creatinine_phosphokinase,yes_no_mapping[diabetes], ejection_fraction,yes_no_mapping[high_blood_pressure],
                platelets,serum_creatinine,serum_sodium,sex_mapping[sex],yes_no_mapping[smoking],time]
  model_input = np.array(input_data).reshape(1,-1)
  prediction = model.predict(model_input)

  if prediction ==1.0:
    return 'Servival chances of patient is low.'
  elif prediction ==0.0:
    return 'Servival chances of patient is high.'
  else:
    return 'Error while making prediction.'


# Inputs from user
inputs = [gradio.Slider(df1['age'].min(), df1['age'].max(), label="Enter the age of the patient:"),
          gradio.Radio(["Yes", "No"], label="Whether patient is Anaemic or not?:"),
          gradio.Slider(round(df['creatinine_phosphokinase'].min(),2), round(df1['creatinine_phosphokinase'].max(),2), label="Enter the level of CPK enzyme in the patient's blood (mcg/L):"),
          gradio.Radio(["Yes", "No"], label="Whether patient is diabetic or not?:"),
          gradio.Slider(df1['ejection_fraction'].min(), df1['ejection_fraction'].max(), label="Enter the % of blood leaving the patient's heart at each contraction:"),
          gradio.Radio(["Yes", "No"], label="Whether patient is Hypertensive or not?:"),
          gradio.Slider(df1['platelets'].min(), df1['platelets'].max(), label="Enter the No. of platelets in the patient's blood (kiloplatelets/mL):"),
          gradio.Slider(round(df1['serum_creatinine'].min(),2), round(df1['serum_creatinine'].max(),2), label="Enter the level of serum creatinine in the patient's blood (mg/dL):"),
          gradio.Slider(round(df1['serum_sodium'].min(),2), round(df1['serum_sodium'].max(),2), label="Enter the level of serum sodium in the patient's blood (mEq/L): "),
          gradio.Radio(["M", "F"], label="Choose the sex of the patient:"),
          gradio.Radio(["Yes", "No"], label="Whether the patient smokes or not?:"),
          gradio.Slider(df1['time'].min(), df1['time'].max(), label="Enter the follow-up period (days):"),
          ]

# Output response
outputs = gradio.Textbox(type="text", label='Will the patient survive?')

# Inputs from user
# YOUR CODE HERE ...

# Output response
# YOUR CODE HERE


# Gradio interface to generate UI link
title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gradio.Interface(fn = predict_death_event,
                         inputs = inputs,
                         outputs = outputs,
                         title = title,
                         description = description,
                         allow_flagging='never')

iface.launch(share = True)  # server_name="0.0.0.0", server_port = 8001   # Ref: https://www.gradio.app/docs/interface

