import numpy as np
import pandas as pd

from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

# Đọc dữ liệu
goddess = pd.read_csv("Goddess.csv")
warrior_2 = pd.read_csv("Warrior_2.csv")
t_pose = pd.read_csv("T_pose.csv")

X = []
y = []
no_of_timesteps = 10

dataset = goddess.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(0)

dataset = warrior_2.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(1)

dataset = t_pose.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(2)

X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model  = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 3, activation="softmax"))
model.compile(optimizer="adam", metrics = ['accuracy'], loss = "categorical_crossentropy")

# Train the model
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)
model.fit(X_train, y_train_categorical, epochs=16, batch_size=32, validation_data=(X_test, y_test_categorical))
model.save("model_4.h5")


