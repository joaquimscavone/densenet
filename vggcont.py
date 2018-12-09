#!/usr/bin/python
# coding:utf-8
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
import setup
import ip
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications import vgg19, vgg16
import json
from keras.callbacks import ModelCheckpoint



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






def create(epochs=250, architecture=19, batch_size=1, MLPinput=4096, MLPhidden=4096, discart_prop=0, optimizer='sgd'):
	img_rows = 224
	img_cols = 224
	channels = 3
	num_classes = 4
	mark = 16; # o número de camadas que devem permanecer congeladas no segundo treinamento
	database = [
    	{"url": 'base/laramin/', "img_type": "jpg", "output":"base/marcacoes.out"}
	]

	

	(train_list, y_train), (test_list, y_test), (valid_list, y_valid) = setup.config_base(database=database, test_prop=0.3, valid_prop=0.1, discart_prop=discart_prop)
	X_train = ip.list_to_array(train_list, (img_rows, img_cols), channels)
	X_test = ip.list_to_array(test_list, (img_rows, img_cols), channels)
	X_valid = ip.list_to_array(valid_list, (img_rows, img_cols), channels)

	#print(X_train.shape)
	#print(X_test.shape)
	#print(X_valid.shape)

	
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)
	y_valid = keras.utils.to_categorical(y_valid, num_classes)

	'''
	print(y_train.shape)
	print(y_test.shape)
	print(y_valid.shape)
	'''

	if architecture == 16:
			model = vgg16.VGG16(include_top=False,weights=None, input_shape=(img_cols, img_rows, channels))
	else:
			model = vgg19.VGG19(include_top=False, weights=None, input_shape=(img_cols, img_rows, channels))
	
		



	

	
	
	fully = model.output
	fully = Flatten()(fully)
	fully = Dense(units=MLPinput, activation='relu', name="MLPInput")(fully)
	#fully = Dropout(dropout)(fully)
	fully = Dense(units=MLPhidden, activation='relu', name="MLPhidden")(fully)
	fully = Dense(units=num_classes, activation='softmax', name="output")(fully)
	model = Model(inputs=model.input,outputs=fully)



	
	
	for layer in model.layers:
		if mark <= 0:
			layer.trainable = True
			#print(layer.name, ' - ativado!')
		else:
			#print(layer.name, ' - desativado!')
			layer.trainable = False
		mark-=1


	model.summary()
	print('Memória usada no modelo:', get_model_memory_usage(batch_size,model))
	

	print("Treinando  com convoluções descongeladas!")


	if optimizer=='sgd':
		model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(), metrics=['accuracy'])
	else:
		model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

	model.load_weights('pesos/t1_best_weights.hdf5')
	
	checkpoint = ModelCheckpoint('pesos/t2_best_weights.hdf5', monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True, mode='max')
	history=model.fit(X_train, y_train,
	          batch_size=batch_size,
	          epochs=epochs,
	          callbacks=[checkpoint,],
	          verbose=1,
	          validation_data=(X_valid, y_valid))
	model.save_weights('pesos/t2_end_weights.hdf5', True)
	file_train_history = open('pesos/t2_history.json', 'w')
	file_train_history.write(json.dumps(history.history))
	file_train_history.close()
	score = model.evaluate(X_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])






	

#create(discart_prop=0.9, batch_size=4)