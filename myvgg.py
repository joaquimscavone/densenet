#!/usr/bin/python
# coding:utf-8
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
#os.environ["CUDA_VISIBLE_DEVICES"]="0" 
import setup
import ip
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications import vgg19, vgg16
import json
from keras.callbacks import ModelCheckpoint


saveweights = False



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



def getTreino():
	arq = open('pesos/treino.txt', 'r')
	texto = arq.readlines()
	arq.close()
	return int(texto[0])
	
def setTreino(treino):
	texto = '%d' % treino
	arq = open('pesos/treino.txt', 'w')
	arq.writelines(texto)
	arq.close()




def create(epochs=250, architecture=19, batch_size=1, MLPinput=4096, MLPhidden=4096, discart_prop=0, convtrain=17, optimizer='sgd'):
	img_rows = 224
	img_cols = 224
	channels = 3
	num_classes = 4
	treinamento = getTreino()+1
	mark = convtrain # 11|17; o número de camadas que devem permanecer congeladas no segundo treinamento
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
			model = vgg16.VGG16(include_top=False,weights='imagenet', input_shape=(img_cols, img_rows, channels))
	else:
			model = vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(img_cols, img_rows, channels))
	
		



	


	for layer in model.layers:
		layer.trainable = False

	
	fully = model.output
	fully = Flatten()(fully)
	fully = Dense(units=MLPinput, activation='relu', name="MLPInput")(fully)
	#fully = Dropout(dropout)(fully)
	fully = Dense(units=MLPhidden, activation='relu', name="MLPhidden")(fully)
	fully = Dense(units=num_classes, activation='softmax', name="output")(fully)
	model = Model(inputs=model.input,outputs=fully)


	model.summary()
	print('Memória usada no modelo:', get_model_memory_usage(batch_size,model))





	if optimizer=='sgd':
		model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(), metrics=['accuracy'])
	else:
		model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])


	print("Treinando fully com convoluções congeladas!")
	
	
	if saveweights :
		checkpoint = ModelCheckpoint('pesos/t%d_f1_best_weights.hdf5'%treinamento, monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=saveweights, mode='max')
		history=model.fit(X_train, y_train,
	    	    			batch_size=batch_size,
	        				epochs=epochs,
	          				callbacks=[checkpoint,],
	          				verbose=1,
	          				validation_data=(X_valid, y_valid))
	
		model.save_weights('pesos/t%d_f1_end_weights.hdf5'%treinamento, True)
	else:
		history=model.fit(X_train, y_train,
	    	    			batch_size=batch_size,
	        				epochs=epochs,
	          				verbose=1,
	          				validation_data=(X_valid, y_valid))
		
	
	file_train_history = open('pesos/t%d_f1_history.json'%treinamento, 'w')
	file_train_history.write(json.dumps(history.history))
	file_train_history.close()
	tinicial = model.evaluate(X_test, y_test, verbose=0)
	print('Test loss:', tinicial[0])
	print('Test accuracy:', tinicial[1])

	for layer in model.layers:
		if mark <= 0:
			layer.trainable = True
			print(layer.name, ' - ativado!')
		else:
			print(layer.name, ' - desativado!')
			layer.trainable = False
		mark-=1
	print('Memória usada no modelo:', get_model_memory_usage(batch_size,model))

	if optimizer=='sgd':
		model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(), metrics=['accuracy'])
	else:
		model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

	if saveweights:
		model.load_weights('pesos/t%d_f1_best_weights.hdf5'%treinamento)
	
	print("Treinando  com convoluções descongeladas!")

	if saveweights:
		checkpoint = ModelCheckpoint('pesos/t%d_f2_best_weights.hdf5'%treinamento, monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True, mode='max')
		history=model.fit(X_train, y_train,
	    	      batch_size=batch_size,
	        	  epochs=epochs,
	          	callbacks=[checkpoint,],
	          	verbose=1,
	          	validation_data=(X_valid, y_valid))
		model.save_weights('pesos/t%d_f2_end_weights.hdf5'%treinamento, True)
	else:
		history=model.fit(X_train, y_train,
	    	      batch_size=batch_size,
	        	  epochs=epochs,
	          	verbose=1,
	          	validation_data=(X_valid, y_valid))
	
	file_train_history = open('pesos/t%d_f2_history.json'%treinamento, 'w')
	file_train_history.write(json.dumps(history.history))
	file_train_history.close()
	tfinal = model.evaluate(X_test, y_test, verbose=0)
	print('Test loss:', tfinal[0])
	print('Test accuracy:', tfinal[1])


	arq = open('pesos/resultados.txt', 'r')
	texto = arq.readlines()
	arq.close()
	texto.append('Treinamento %d---------------------------------------\n'%treinamento)
	texto.append('epochs=%d\nMLPinput=%d\nMLPhidden=%d\noptimizer=%s\nconvtrain=%d\n' % (epochs,MLPinput,MLPhidden,optimizer,convtrain))
	texto.append('Treinamento inicial:\n')
	texto.append('Test loss: %f \n' % tinicial[0])
	texto.append('Test accuracy: %f \n' %tinicial[1])
	texto.append('Treinamento Final:\n')
	texto.append('Test loss: %f \n' % tfinal[0])
	texto.append('Test accuracy:%f \n' % tfinal[1])
	texto.append('Fim do treinamento %d---------------------------------------\n\n\n'%treinamento)
	arq = open('pesos/resultados.txt', 'w')
	arq.writelines(texto)
	arq.close()
	setTreino(treinamento)
	del model
	keras.backend.clear_session()
	return tfinal[1]


def hyper(params):
	epochs=params['epochs']
	#epochs = 1
	MLPinput=params['MLPhidden']
	MLPhidden=params['MLPhidden']
	optimizer=params['optimizer']
	convtrain=params['convtrain']
	batch_size=params['batch_size']
	return 1 - create(epochs=epochs, MLPinput=MLPinput, MLPhidden=MLPhidden, optimizer=optimizer,convtrain=convtrain, batch_size=batch_size, discart_prop=0)





#create(discart_prop=0.999, batch_size=1, epochs=1)