'''
Tensorflow: opem source library to help develop and train machine learning models

Example of a neural net by using tensorflows

'''

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# these are features that allow the model to identify the class- this is an example on supervised learning
col_names = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym","fM3Long", "fM3Trans", "fAlpha", "fDist","class"]
csv_data = pd.read_csv("magic+gamma+telescope/magic04.data", names=col_names)

csv_data["class"] = (csv_data["class"] == "g").astype(int)

def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_loss(history):
    plt.plot(history.history['loss'], label= 'loss')
    plt.plot(history.history['val_loss'], label= 'val_loss')
    plt.xlabel("Epoch")
    plt.ylabel("Binary crossentropy")
    plt.legend()
    plt.grid(True)
    plt.show()
# creating train, validation, and test datasets

# split array into subarrays for training, validation, and testing data sets
# note that this method of splitting is deprecated and will be removed at some point- look into iloc instead
train, valid, test = np.split(csv_data.sample(frac=1), [int(0.6 * len(csv_data)), int(0.8 * len(csv_data))])
#create_hist_head()

'''
scale_dataset() takes dataframe and an oversample boolean
as arguments


Scale dataset returns data as a horizontal stack, along with it's respective x and y values (equal amounts of x/y values )

'''
def scale_dataset(dataframe, oversample=False):
    x_values = dataframe[dataframe.columns[:-1]].values
    y_values = dataframe[dataframe.columns[-1]].values
    
    scaler = StandardScaler()
    x_values = scaler.fit_transform(x_values)

    if oversample:
        ros = RandomOverSampler()
        x_values, y_values = ros.fit_resample(x_values, y_values)

    data = np.hstack((x_values, np.reshape(y_values, (-1,1))))

    return data, x_values, y_values

# datasets finally prepared!
train, x_train, y_train = scale_dataset(train, oversample=True)
valid, x_valid, y_valid = scale_dataset(valid, oversample=False)
test, x_test, y_test = scale_dataset(test, oversample= False)

# creating neural network to train the models
nn_model = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu', input_shape = (10,)), 
                                tf.keras.layers.Dense(32, activation='relu'),
                                tf.keras.layers.Dense(1, activation='sigmoid')])

nn_model.compile(optimizer= tf.keras.optimizers.Adam(0.001), loss="binary_crossentropy",
                 metrics=['accuracy'])

history = nn_model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

plot_accuracy(history)
plot_loss(history)
