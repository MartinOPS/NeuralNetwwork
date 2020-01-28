import warnings
warnings.filterwarnings("ignore")
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



dataset = pd.read_csv('0115_avila.csv').dropna()

intercolumnarDistance = dataset.iloc[:, 2].values
upperMargin = dataset.iloc[:, 3].values
lowerMargin = dataset.iloc[:, 4].values
exploitation = dataset.iloc[:, 5].values
rowNumber = dataset.iloc[:, 6].values
modularRatio = dataset.iloc[:, 7].values
interlinearSpacing = dataset.iloc[:, 8].values
weight = dataset.iloc[:, 9].values
peakNumber = dataset.iloc[:, 10].values
yvalue_class = dataset.iloc[:, 11].values #Y value, Vectorize

encoder = LabelEncoder()


y = encoder.fit_transform(yvalue_class)
Y = pd.get_dummies(y).values
Y = np.array(Y)

X = []
for a,b,c,d,e,f,g,h,i in zip(intercolumnarDistance, upperMargin, lowerMargin, exploitation, rowNumber, modularRatio, intercolumnarDistance, weight, peakNumber):
    X.append([a,b,c,d,e,f,g,h,i])
X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y)

model = Sequential()

model.add(Dense(8, input_shape=(9, ), activation='softmax'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(12, activation='relu'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=20)

y_pred = model.predict(X_test)

y_test_class = np.argmax(y_test,axis=1)
y_pred_class = np.argmax(y_pred,axis=1)


history1 = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=20)

print(classification_report(y_test_class, y_pred_class))
a = accuracy_score(y_pred_class,y_test_class)
print('Accuracy is:',a*20)

print(confusion_matrix(y_test_class, y_pred_class))

plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('Presnosť')
plt.ylabel('Presnosť')
plt.xlabel('Epocha')
plt.legend(['Trénovacia M', 'Testovacia M'], loc='upper left')
plt.show()

plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('Chyba')
plt.ylabel('Strata')
plt.xlabel('Epocha')
plt.legend(['Trénovacia M', 'Testovacia M'], loc='upper left')
plt.show()