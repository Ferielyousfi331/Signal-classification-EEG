import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Convolution1D, MaxPool1D, Flatten, Dropout
from keras.layers import Input, LSTM
from keras.models import Model
from tensorflow.keras import layers
from keras.layers import BatchNormalization
import keras
from keras.callbacks import ModelCheckpoint

# Load the raw data from a CSV file
raw_data = pd.read_csv("data.csv")
raw_data.head()
raw_data.tail()

# Extract relevant data columns
data = raw_data.values
data = data[1:11501, 1:180]
data.shape

# Create DataFrames for each category (1-5)
D = data
df_1 = D[D[:, 178]==1]
df_2 = D[D[:, 178]==2]
df_3 = D[D[:, 178]==3]
df_4 = D[D[:, 178]==4]
df_5 = D[D[:, 178]==5]

# Display the shape of each category
print(df_1.shape)
print(df_2.shape)
print(df_3.shape)
print(df_4.shape)
print(df_5.shape)

# Convert DataFrames to integers
df_1 = df_1.astype(int)
df_2 = df_2.astype(int)
df_3 = df_3.astype(int)
df_4 = df_4.astype(int)
df_5 = df_5.astype(int)

# Adjust the labels for categories 2-5
df_2[:, 178] = df_2[:, 178] - 2
df_3[:, 178] = df_3[:, 178] - 3
df_4[:, 178] = df_4[:, 178] - 4
df_5[:, 178] = df_5[:, 178] - 5

# Concatenate categories 1 and 3 into a new dataset (D1)
D1 = np.concatenate([df_1, df_3])

# Randomly split D1 into training, validation, and test sets
number_of_rows = D1.shape[0]
random_indices = np.random.choice(number_of_rows, size=int(number_of_rows*0.8), replace=False)

label_train = D1[random_indices, -1]
data_train = D1[random_indices, :-1]

D1_rest = np.delete(D1, random_indices, 0)

number_of_rows = D1_rest.shape[0]
random_indices = np.random.choice(number_of_rows, size=int(number_of_rows*0.5), replace=False)

label_val = D1_rest[random_indices, -1]
data_val = D1_rest[random_indices, :-1]

D1_rest_rest = np.delete(D1_rest, random_indices, 0)
label_test = D1_rest_rest[:, -1]
data_test = D1_rest_rest[:, :-1]

# Expand dimensions for Conv1D input
data_train = np.expand_dims(data_train, axis=2)
data_val = np.expand_dims(data_val, axis=2)
data_test = np.expand_dims(data_test, axis=2)

# Display shapes of training, validation, and test sets
print(label_train.shape, data_train.shape)
print(label_val.shape, data_val.shape)
print(label_test.shape, data_test.shape)

# Function to evaluate and plot the performance of a model
def evaluate_model(history, X_test, y_test, model, model_name):
    # Evaluate the model on the test set
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f"Accuracy for {model_name}: {scores[1]*100:.2f}%")

    # Plot accuracy and loss curves
    fig1, ax_acc = plt.subplots()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()

    fig2, ax_loss = plt.subplots()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()

    # Display confusion matrix
    target_names = ['1', '2', '3']
    y_true = [np.argmax(element) for element in y_test]
    prediction_proba = model.predict(X_test)
    prediction = np.argmax(prediction_proba, axis=1)
    cnf_matrix = confusion_matrix(y_true, prediction)
    print(f"Confusion Matrix for {model_name}:\n{cnf_matrix}")

# Function to save the model
def save_model(model, model_name):
    model.save(f"{model_name}.h5")
    print(f"Model {model_name} saved successfully!")

# Define CNN network
def network_CNN(X_train, y_train):
    im_shape=(X_train.shape[1], 1)
    inputs_cnn=Input(shape=(im_shape), name='inputs_cnn')
    conv1d_1 = layers.Conv1D(filters=32, kernel_size=6)(inputs_cnn)
    batch_normalization = BatchNormalization()(conv1d_1)
    max_pooling1d = layers.MaxPooling1D(2, padding='same')(batch_normalization)
    conv1d_2 = layers.Conv1D(filters=64, kernel_size=3)(max_pooling1d)
    batch_normalization_1 = BatchNormalization()(conv1d_2)
    max_pooling1d_1 = layers.MaxPooling1D(2, padding='same')(batch_normalization_1)
    flatten = Flatten()(max_pooling1d_1)
    dense = Dense(32)(flatten)
    dense_1 = Dense(16)(dense)
    main_output = Dense(2)(dense_1)
    model1 = Model(inputs=inputs_cnn, outputs=main_output)
    model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model1

# Train and evaluate the first CNN model
model1 = network_CNN(data_train, label_train)
print(model1.summary())
save_path_model1 = 'model_checkpoints/checkpoint_1'
model_checkpoint_callback = ModelCheckpoint(
    filepath=save_path_model1,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

history_model1 = model1.fit(data_train, label_train, epochs=100, batch_size=32,
                            validation_data=(data_val, label_val),
                            callbacks=[model_checkpoint_callback])

evaluate_model(history_model1, data_test, label_test, model1, 'Model 1')
save_model(model1, 'Model_1')

# Define Bidirectional LSTM model
def network_LSTM(X_train, y_train):
    im_shape=(X_train.shape[1], 1)
    inputs_lstm=Input(shape=(im_shape), name='inputs_lstm')

    dense = Dense(units=32, activation='relu', name='dense')(inputs_lstm)
    lstm = layers.Bidirectional(LSTM(units=128, name='lstm'))(dense)
    dropout = Dropout(0.3)(lstm)
    batch_normalization = BatchNormalization(name='batch_normalization')(dropout)
    dense_1 = Dense(units=64, activation='relu', name='dense_1')(batch_normalization)
    dropout_2 = Dropout(0.3, name='dropout_2')(dense_1)
    batch_normalization_1 = BatchNormalization(name='batch_normalization_1')(dropout_2)
    main_output = Dense(units=2, activation='softmax')(batch_normalization_1)

    model = Model(inputs=inputs_lstm, outputs=main_output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Train and evaluate the first Bidirectional LSTM model
model2 = network_LSTM(data_train, label_train)
print(model2.summary())
save_path_model2 = 'model_checkpoints/checkpoint_2'
model_checkpoint_callback = ModelCheckpoint(
    filepath=save_path_model2,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

history_model2 = model2.fit(data_train, label_train, epochs=100, batch_size=32,
                            validation_data=(data_val, label_val),
                            callbacks=[model_checkpoint_callback])

evaluate_model(history_model2, data_test, label_test, model2, 'Model 2')
save_model(model2, 'Model_2')

# Define the data categories: Comparing CNN and BiLSTM on Epilepsy versus All Other Data Categories
D = data
df_1 = D[D[:, 178]==1]
df_2 = D[D[:, 178]==2]
df_3 = D[D[:, 178]==3]
df_4 = D[D[:, 178]==4]
df_5 = D[D[:, 178]==5]

# Display the shape of each category
print(df_1.shape)
print(df_2.shape)
print(df_3.shape)
print(df_4.shape)
print(df_5.shape)

# Convert DataFrames to integers
df_1 = df_1.astype(int)
df_2 = df_2.astype(int)
df_3 = df_3.astype(int)
df_4 = df_4.astype(int)
df_5 = df_5.astype(int)

# Concatenate 4 categories into 1 data set
df_2[:, 178] = df_2[:, 178] - 2
df_3[:, 178] = df_3[:, 178] - 3
df_4[:, 178] = df_4[:, 178] - 4
df_5[:, 178] = df_5[:, 178] - 5

D2 = np.concatenate([df_1, df_2, df_3, df_4, df_5])

# Randomly split D2 into training, validation, and test sets
number_of_rows = D2.shape[0]
random_indices = np.random.choice(number_of_rows, size=int(number_of_rows*0.8), replace=False)

label_train_all = D2[random_indices, -1]
data_train_all = D2[random_indices, :-1]

D2_rest = np.delete(D2, random_indices, 0)

number_of_rows = D2_rest.shape[0]
random_indices = np.random.choice(number_of_rows, size=int(number_of_rows*0.5), replace=False)

label_val_all = D2_rest[random_indices, -1]
data_val_all = D2_rest[random_indices, :-1]

D2_rest_rest = np.delete(D2_rest, random_indices, 0)
label_test_all = D2_rest_rest[:, -1]
data_test_all = D2_rest_rest[:, :-1]

# Expand dimensions for Conv1D input
data_train_all = np.expand_dims(data_train_all, axis=2)
data_val_all = np.expand_dims(data_val_all, axis=2)
data_test_all = np.expand_dims(data_test_all, axis=2)

# Display shapes of training, validation, and test sets
print(label_train_all.shape, data_train_all.shape)
print(label_val_all.shape, data_val_all.shape)
print(label_test_all.shape, data_test_all.shape)

# Define CNN model to be trained on epileptic vs all data
model3 = network_CNN(data_train_all, label_train_all)
print(model3.summary())
save_path_model3 = 'model_checkpoints/checkpoint_3'
model_checkpoint_callback = ModelCheckpoint(
    filepath=save_path_model3,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

history_model3 = model3.fit(data_train_all, label_train_all, epochs=100, batch_size=32,
                            validation_data=(data_val_all, label_val_all),
                            callbacks=[model_checkpoint_callback])

# Print validation accuracy and plot accuracy and loss
evaluate_model(history_model3, data_test_all, label_test_all, model3, 'Model 3')
save_model(model3, 'Model_3')

# Define Bidirectional LSTM model to be trained on epileptic vs all data
model4 = network_LSTM(data_train_all, label_train_all)
print(model4.summary())
save_path_model4 = 'model_checkpoints/checkpoint_4'
model_checkpoint_callback = ModelCheckpoint(
    filepath=save_path_model4,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

history_model4 = model4.fit(data_train_all, label_train_all, epochs=100, batch_size=32,
                            validation_data=(data_val_all, label_val_all),
                            callbacks=[model_checkpoint_callback])

# Print validation accuracy and plot accuracy and loss
evaluate_model(history_model4, data_test_all, label_test_all, model4, 'Model 4')
save_model(model4, 'Model_4')
