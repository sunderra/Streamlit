import keras
'''
Created on 24 Sept 2024
Name     : Sunder Rajan
Type     : Forex Hedged USD
Server   : MetaQuotes-Demo
Login    : 86791898
Password : 4+OwMhRv
Investor : WaV+Z4Bo

@author: sunde
'''

from keras.models import Sequential
from keras.layers import Dense
#import keras
import pandas as pd

import pandas_datareader as pdr
import matplotlib.pyplot as plt
import numpy as np
from master_function import data_preprocessing, mass_import
from master_function import plot_train_test_values
from master_function import calculate_accuracy,model_bias, RMSE
from sklearn.metrics import mean_squared_error


start_date = "1990-01-01"
end_date   = "2023-06-01"
data = np.array((pdr.get_data_fred('SP500',start=start_date,end=end_date)).dropna())
data = np.diff(data[:,0])

num_lags = 100  # number of lagged values we use for prediction
train_test_split = 0.80
num_neurons_in_hidden_layers = 20
num_epochs = 100
batch_size=16

x_train,y_train,x_test, y_test = data_preprocessing(data,num_lags, train_test_split)

# now fit the model
model = Sequential()
#first hidden layer with input and ReLU activation function
model.add(Dense(num_neurons_in_hidden_layers,input_dim=num_lags, activation='relu'))

model.add(Dense(num_neurons_in_hidden_layers, activation='relu'))
# output layer
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')


losses = []
epochs = []

import tensorflow as tf
from keras.callbacks import Callback

class LossCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        losses.append(logs['loss'])
        epochs.append(epoch+1)
        plt.clf()
        plt.plot(epochs, losses, marker='o')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.grid(True)
        plt.pause(0.01)

model.fit(x_train,np.reshape(y_train,(-1,1)), epochs=num_epochs, verbose=0, batch_size=batch_size, callbacks=[LossCallback()])    
    
y_predicted_train = np.reshape(model.predict(x_train),(-1,1))
y_predicted       = np.reshape(model.predict(x_test),(-1,1))

plot_train_test_values(100,50,y_train,y_test,y_predicted)

acc_train = calculate_accuracy(y_predicted_train, y_train)
acc_test  = calculate_accuracy(y_predicted, y_test)
print("Train accuracy: " +str(acc_train) + ", Test accuracey: " + str(acc_test))

mb_train = model_bias(y_predicted_train)
mb_test = model_bias(y_predicted)
print("Bias train: " + str(mb_train) +", test bias: " + str(mb_test))

print("RSME test:" + str(RMSE(y_predicted,np.reshape(y_test,(-1,1)))))
print("RSME train:" + str(RMSE(y_predicted_train,np.reshape(y_train,(-1,1)))))

# correlation
test_corr = pd.DataFrame({'col1': y_predicted[:,0], 'col2': y_test}).corr()
print("Test correlation " + str(test_corr))

train_corr = pd.DataFrame({'col1': y_predicted_train[:,0], 'col2': y_train}).corr()
print("Train correlation " + str(train_corr))    

