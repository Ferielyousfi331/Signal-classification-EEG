import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Dropout, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix, accuracy_score
from sklearn.utils import class_weight
from keras.callbacks import ModelCheckpoint

# Load data
data = pd.read_csv('data.csv')

# Process data
X = np.asarray(data.values)
X = np.asarray(X[:, 1:-1])
X = X.astype(float)
minm = X.min()
maxm = X.max()
X_norm = (X - minm) / (maxm - minm)
X = X.reshape(11500, 178, 1)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
plt.plot(X[1,:],label='1')
plt.plot(X[7,:],label='2')
plt.plot(X[12,:],label='3')
plt.plot(X[0,:],label='4')
plt.plot(X[2,:],label='5')
plt.legend()
plt.show()


mlb = LabelBinarizer()
y = np.asarray(data['y'])
Y = y.astype(float) - 1
Y = to_categorical(Y)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=1)

# Define the model
model = Sequential()
model.add(Conv1D(filters=2048, kernel_size=3, activation='relu', input_shape=(178, 1)))
model.add(Dropout(0.5))
model.add(Conv1D(filters=1024, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(Conv1D(filters=512, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
model.add(Dropout(0.3))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(Dropout(0.3))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(Dropout(0.3))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.3))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(5, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Compute class weights
y_ints = [y.argmax() for y in Y_train]
unique_classes = np.unique(y_ints)
class_weights = len(y_ints) / (len(unique_classes) * np.bincount(y_ints))
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Define callbacks
mcp_save = ModelCheckpoint('cnn_epil.hdf5', save_best_only=True, verbose=1, monitor='val_accuracy', mode='max')

# Train the model
model.fit(X_train, Y_train, epochs=70, batch_size=100, verbose=1, callbacks=[mcp_save], validation_split=0.2, class_weight=class_weight_dict)

# Evaluate the model
scores = model.evaluate(X_test, Y_test, verbose=0, batch_size=200)
pred = model.predict(X_test, verbose=0, batch_size=200)
pred = np.round(pred)
print(classification_report(Y_test, pred))
f1 = precision_recall_fscore_support(Y_test, pred, average='micro')

# Calculate and print accuracy
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# Make predictions
ypred = model.predict((X_test[:, ::4] - X_test.mean()) / X_test.std())
ypred.shape

yp = np.zeros((Y_test.shape[0]))
yo = np.ones((Y_test.shape[0]))

for i in range(Y_test.shape[0]):
    yp[i] = np.argmax(ypred[i]) + 1
    yo[i] = np.argmax(Y_test[i])

yp.shape
yo.shape

np.unique(yo)
np.unique(Y_test)
np.unique(yp)

yo.shape

for i in range(Y_test.shape[0]):
    if yo[i] != 1:
        yo[i] = 0
    if yp[i] != 1:
        yp[i] = 0

np.unique(yo)
np.unique(yp)

# Calculate accuracy using sklearn
accuracy_score(yo, yp)

