import setup
import ip
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


database = [
    {"url": 'Digitos/', "img_type": "jpg"}
]

(train_list, y_train), (test_list, y_test), (valid_list, y_valid) = setup.config_base(database=database, test_prop=0.3, valid_prop=0.1)

img_rows = 32
img_cols = 32
channels = 3

X_train = ip.list_to_array(train_list, (img_rows, img_cols), channels)
X_test = ip.list_to_array(test_list, (img_rows, img_cols), channels)
X_valid = ip.list_to_array(valid_list, (img_rows, img_cols), channels)

print(X_train.shape)
print(X_test.shape)
print(X_valid.shape)

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, channels)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])

print("Modelo compilado!")

model.fit(X_train, y_train,
          batch_size=128,
          epochs=250,
          verbose=1,
          validation_data=(X_valid, y_valid))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
