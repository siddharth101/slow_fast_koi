from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D
from keras.regularizers import l2
from keras.layers import Dense, Dropout, Activation, Flatten

input_shape = (140, 170, 1)
W_reg = 1e-4

model = Sequential()
model.add(Conv2D(16, (5, 5), padding='valid',
              input_shape=input_shape,
              kernel_regularizer=l2(W_reg)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))


model.add(Conv2D(32, (5, 5), padding='valid', kernel_regularizer=l2(W_reg)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (5, 5), padding='valid', kernel_regularizer=l2(W_reg)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (5, 5), padding='valid', kernel_regularizer=l2(W_reg)))
model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, kernel_regularizer=l2(W_reg)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
print (model.summary())

model.save('gravityspy_model.h5')
