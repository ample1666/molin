import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import model_from_json


# 准备好数据
data = pd.read_csv("hellobike.csv")
data = np.array(data)
X = data[:, 1:]
y = data[:, 0]
y = y.reshape(839604,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# 模型构建
model = Sequential()
model.add(Dense(input_dim=61, units=12, kernel_initializer='uniform', activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=10, kernel_initializer='uniform', activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# 模型训练和评估
model.fit(X_train, y_train, batch_size=128, epochs=2)
result_train = model.evaluate(X_train, y_train)
result_test = model.evaluate(X_test, y_test)
print("training acc: %f" % result_train[1])
print("testing acc: %f" % result_test[1])