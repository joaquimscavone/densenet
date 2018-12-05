#!/usr/bin/python
# coding:utf-8
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
import setup
import ip
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications import densenet


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


database = [
    {"url": 'base/lara/', "img_type": "jpg"}
]





discart_prop = 0.99
batch_size=1
epochs=5
arquitetura = 1



(train_list, y_train), (test_list, y_test), (valid_list, y_valid) = setup.config_base(database=database, test_prop=0.3, valid_prop=0.1, discart_prop=discart_prop)


img_rows = 480
img_cols = 640
channels = 3


X_train = ip.list_to_array(train_list, (img_rows, img_cols), channels)
X_test = ip.list_to_array(test_list, (img_rows, img_cols), channels)
X_valid = ip.list_to_array(valid_list, (img_rows, img_cols), channels)


print(X_train.shape)
print(X_test.shape)
print(X_valid.shape)

num_classes = 4
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)


print(y_train.shape)
print(y_test.shape)
print(y_valid.shape)


if arquitetura == 1:
	dense = densenet.DenseNet169(include_top=True, weights=None, input_shape=(img_cols, img_rows, channels), classes=num_classes)
elif arquitetura == 2:
	dense = densenet.DenseNet121(include_top=True, weights=None, input_shape=(img_cols, img_rows, channels), classes=num_classes)
else:
	dense = densenet.DenseNet201(include_top=True, weights=None, input_shape=(img_cols, img_rows, channels), classes=num_classes)



dense.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])




dense.summary()


#print(get_model_memory_usage(1,dense))

print("Modelo compilado!")


dense.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(X_valid, y_valid))

score = dense.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])