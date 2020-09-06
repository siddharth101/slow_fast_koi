import pickle
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam



import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.95):
      print("\nReached 95% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

model = tf.keras.models.Sequential([
    # First Conv layer
    tf.keras.layers.Conv2D(64,(3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Second Conv layer
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Third Conv layer
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Fourth Conv Layer
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    #Flatten
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),

    # Dense Layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')])

model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])
model.summary()

TRAINING_DIR = '/home/siddharth.soni/TensorFlow_fast_slow/Training/' 
VALIDATION_DIR =  '/home/siddharth.soni/TensorFlow_fast_slow/Validation/'

train_datagen = ImageDataGenerator(rescale=1.0/255.0)
validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_datagenerator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                      batch_size=20,
                                                      target_size=(150,150),
                                                      class_mode='categorical')

validation_datagenerator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                      batch_size=20,
                                                      target_size=(150,150),
                                                      class_mode='categorical')

history = model.fit(train_datagenerator, steps_per_epoch= 765//20,
                    epochs=10, verbose=True, validation_data=validation_datagenerator,
                    validation_steps= 145//20, callbacks = [callbacks])
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

model.save('/home/siddharth.soni/TensorFlow_fast_slow/fast_slow_koi.h5')

f = open('history.pkl', 'wb')
pickle.dump(history.history, f)
f.close()
