# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
This dataset presents a captivating challenge due to the intricate relationship between the input and output columns. The complex nature of this connection suggests that there may be underlying patterns or hidden factors that are not readily apparent.

## Neural Network Model

![dl_e1_diagram](https://github.com/Ronick2005/basic-nn-model/assets/83219341/40de5741-1d0d-4ce9-97dd-5c520879a4d1)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Ronick Aakshath P
### Register Number: 212222240084
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('exp1').sheet1

rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'string'})
df = df.astype({'Output':'string'})
df.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

le = LabelEncoder()
df["Input"] = le.fit_transform(df["Input"])
df["Output"] = le.fit_transform(df["Output"])

X = df[['Input']].values
y = df[['Output']].values

X

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

Scaler = MinMaxScaler()

X_train1 = Scaler.fit_transform(X_train)

ai_brain = Sequential()

X_train.shape

ai_brain.add(Dense(5, activation = "relu", input_shape = X_train.shape))
ai_brain.add(Dense(10, activation = "relu"))
ai_brain.add(Dense(1))

ai_brain.summary()

ai_brain.compile(optimizer = 'sgd', loss = 'mse')

ai_brain.fit(X_train1, y_train, epochs = 100)

ai_brain.history

loss_df = pd.DataFrame(ai_brain.history.history)

import matplotlib.pyplot as plt

plt.title("loss curve")
plt.plot(loss_df)

X_test1 = Scaler.transform(X_test)

ai_brain.evaluate(X_test1,y_test)

X_n1 = [[30]]

X_n1_1 = Scaler.transform(X_n1)

ai_brain.predict(X_n1_1)
```
## Dataset Information
![image](https://github.com/Ronick2005/basic-nn-model/assets/83219341/886af0c5-f89d-4db0-aaec-f94a22d3b8d6)

## OUTPUT

### Training Loss Vs Iteration Plot
![image](https://github.com/Ronick2005/basic-nn-model/assets/83219341/68632f48-77d7-443b-a127-d151b6b9a308)

### Test Data Root Mean Squared Error
![image](https://github.com/Ronick2005/basic-nn-model/assets/83219341/dd6070f4-2501-4c69-9f73-676e17b7e27d)

### New Sample Data Prediction
![image](https://github.com/Ronick2005/basic-nn-model/assets/83219341/a9d0e3d7-c8c1-4312-b15c-9ba5e75583e3)

## RESULT
Summarize the overall performance of the model based on the evaluation metrics obtained from testing data as a regressive neural network based prediction has been obtained.
