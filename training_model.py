import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

'''
This script is to be run only one time to generate the model that detect the emotions, and each time we will use that generated
model (emotion_model.h5) to detect the emotions.
'''

def load_and_process_data(file_path="fer2013.csv"):
    '''
    This function loads the FER-2013 dataset from a CSV file, and then process the data to
    prepare it for the model for training and testing
    :param file_path: fer2013.csv
    :return: x_train, x_test, y_train, y_test
    '''
    #Reads the FER-2013 dataset, where each row contains pixel data and an emotion label.
    data = pd.read_csv(file_path)
    #Stores image pixel data in x, and emotion labels in y.
    x = []
    y = []
    #Iterate over each row in the dataset.
    for _, row in data.iterrows():
        #Convert pixels into a NumPy array, and reshape the 1D array into a 48x48 grayscale image with a single channel (1).
        pixels = np.array(row['pixels'].split(), dtype = 'float32').reshape(48, 48, 1)
        x.append(pixels) #Appends the reshaped image to x.
        y.append(row['emotion']) #Appends the emotion label to y.

    x = np.array(x) / 255.0 #Converts the list of images to a NumPy array, and then Normalizes pixel values to the range [0,1].
    y = to_categorical(np.array(y), num_classes=7) #Converts the list of labels to a NumPy array, and converts the labels into hard-coded vectors with 7 classes, since there are 7 emotions.
    #Splits the data into 80% training and 20% testing with a random state.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test

def build_model():
    '''
    This function creates the architecture of the Convolutional Neural Network (CNN) for emotion detection.
    Returns a compiled CNN model ready for training.
    :return: model
    '''
    #Initializes a linear stack of layers for the CNN.
    model = Sequential()
    #Adds a convolutional layer with 64 filters, each of size 3x3. Uses the ReLU activation to introduce non_linearity. Specifies the input dimensions which is grayscale 48x48 images.
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2))) #Pooling reduces the dimensions by taking the max value in each 2x2 block.
    #Second convolutional layer extracts more complex features using 128 filters
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Third convolutional layer detects even higher level features using 256 features.
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Converts the 3D outputs from the convolutional layers into a 1D vector.
    model.add(Flatten())
    #Adds a fully connected layer with 512 neurons.
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5)) #Randomly sets 50% of the neurons to zero during training to prevent overfitting.
    model.add(Dense(7, activation='softmax')) #7 neurons for the 7 emotion classes. Uses softmax to compute class probabilities.
    #Configures the model for training. The adam optimizer is an adaptive moment estimation optimizer. The categorical_crossentropy is a loss function for multi-class classification. Metrics is to track the accuracy during training
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(x_train, x_test, y_train, y_test):
    '''
    This function trains the CNN model using the processed data, saves the trained model to a file of type .h5 for later use and to
    and returns the history which contains metrics such as training accuracy, validation accuracy, training loss, and validation loss for each epoch.
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return: history
    '''
    model = build_model() #Initializes the CNN model
    #contian the training and the validation data, epoches is the number of times the model sees the full training data, the batch size is the number of samples per training steps, and the verbose displays the progress.
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30, batch_size=64, verbose=1)
    model.save("emotion_model.h5")
    print("Model saved as emotion_model.h5")
    return history

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_and_process_data("fer2013.csv")
    history = train_model(x_train, x_test, y_train, y_test)


