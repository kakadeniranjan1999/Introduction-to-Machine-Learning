import matplotlib.pyplot as plt
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# All the files related to this model will be saved with this name
filename = 'pima-indian-diabetes-dataset'

# load the dataset
dataset = loadtxt('datasets/pima-indian-diabetes-dataset/pima-indians-diabetes-dataset.csv', delimiter=',')

# split into input (X) and output (y) variables
X = dataset[:, 0:8]
y = dataset[:, 8]

# Split data into training and testing set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu', kernel_regularizer='L1L2'))
model.add(Dense(8, activation='relu', kernel_regularizer='L1L2'))
model.add(Dense(1, activation='sigmoid', kernel_regularizer='L1L2'))

# Print model summary
model.summary()

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=400, batch_size=16)

# Plot accuracy and loss plots
# Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('TrainingData/Accuracy/' + filename + '.png')
plt.show()

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig('TrainingData/Loss/' + filename + '.png')
plt.show()

# evaluate the keras model
metrics = model.evaluate(x_test, y_test)
print('Accuracy of the model is {}'.format(metrics[1] * 100))

# save model
model.save('TrainedModels/' + filename + '.h5')
